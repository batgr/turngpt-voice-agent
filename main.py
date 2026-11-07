"""Voice agent: microphone -> VAD -> Whisper -> TurnGPT decides when the user is done -> LLM -> Kokoro.
Use a headset: on loudspeakers the agent can hear itself."""

import torch

from agent.agent import Agent
from speech.audio_recorder import AudioRecorder
from speech.speech_to_text import SpeechToText
from speech.text_to_speech import TextToSpeech
from speech.voice_activity_detection import VoiceActivityDetector
from turn.turn_gpt import TurnGPT
from turn.turn_taking import TurnTaking


def run(chunks, vad, stt, turn_gpt, turns, agent, tts):
    """chunks: 32 ms of 16 kHz mono audio, as numpy arrays"""
    text = ""
    for chunk in chunks:
        wave = torch.from_numpy(chunk).view(1, -1)
        is_speech = vad.is_speaking(wave)
        if is_speech or vad.buffer:
            vad.add_to_buffer(wave)

        step = turns.update(is_speech, tts.speaking)
        if step == "discard":
            vad.buffer.clear()
        elif step == "interrupt":
            tts.stop()
        elif step == "ask":
            text = stt.to_text(vad.get_full_audio())
            turns.turn_shift = turn_gpt.turn_shift_probability(agent.turns() + [text]) if text else 0.0
            print(f"> {text} (P(turn shift) = {turns.turn_shift:.2f})")
        elif step == "answer":
            turns.reset()
            vad.reset()
            if text:
                reply = agent.response(text)
                print(reply)
                tts.to_speech(reply)


if __name__ == "__main__":
    recorder = AudioRecorder()
    components = VoiceActivityDetector(), SpeechToText(), TurnGPT(), TurnTaking(), Agent(), TextToSpeech()
    recorder.rec()
    print("Listening. Ctrl-C to stop.")
    try:
        run(recorder.chunks(), *components)
    except KeyboardInterrupt:
        print("Stop")
