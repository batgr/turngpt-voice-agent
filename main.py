"""Voice agent: microphone -> VAD -> Whisper -> TurnGPT decides when the user is done -> LLM -> Kokoro.
With a headset, speaking cuts the agent off. On loudspeakers, use --speakers: the agent would hear itself."""

import argparse
from datetime import datetime

import sounddevice as sd
import torch

from agent.agent import Agent
from speech.audio_recorder import AudioRecorder
from speech.speech_to_text import SpeechToText
from speech.text_to_speech import TextToSpeech
from speech.voice_activity_detection import VoiceActivityDetector
from turn.turn_gpt import TurnGPT
from turn.turn_taking import TurnTaking

CHUNKS_PER_LEVEL_LOG = 64  # about 2 s


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def run(chunks, vad, stt, turn_gpt, turns, agent, tts):
    """chunks: 32 ms of 16 kHz mono audio, as numpy arrays"""
    text = ""
    peak = 0.0
    for i, chunk in enumerate(chunks, start=1):
        peak = max(peak, float(abs(chunk).max()))
        if i % CHUNKS_PER_LEVEL_LOG == 0:
            log(f"mic level {peak:.3f}")  # 0.000: no sound reaches the program
            peak = 0.0

        wave = torch.from_numpy(chunk).view(1, -1)
        is_speech = vad.is_speaking(wave)
        if is_speech or vad.buffer:
            vad.add_to_buffer(wave)

        step = turns.update(is_speech, tts.speaking)
        if is_speech and turns.speech == 1:
            log("speech started")
        if step == "discard":
            log("too short or agent speaking: ignored")
            vad.buffer.clear()
        elif step == "interrupt":
            log("agent interrupted")
            tts.stop()
        elif step == "ask":
            text = stt.to_text(vad.get_full_audio())
            turns.turn_shift = turn_gpt.turn_shift_probability(agent.turns() + [text]) if text else 0.0
            log(f"user: {text}")
            log(f"P(turn shift) = {turns.turn_shift:.2f}")
        elif step == "answer":
            turns.reset()
            vad.reset()
            if text:
                reply = agent.response(text)
                log(f"agent: {reply}")
                tts.to_speech(reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", action="store_true", help="ignore the microphone while the agent talks")
    args = parser.parse_args()

    log(f"input: {sd.query_devices(kind='input')['name']}, output: {sd.query_devices(kind='output')['name']}")
    recorder = AudioRecorder()
    turns = TurnTaking(barge_in=3600 if args.speakers else 0.3)  # an hour of speech: never
    components = VoiceActivityDetector(), SpeechToText(), TurnGPT(), turns, Agent(), TextToSpeech()
    recorder.rec()
    log("listening, Ctrl-C to stop")
    try:
        run(recorder.chunks(), *components)
    except KeyboardInterrupt:
        log("stop")
