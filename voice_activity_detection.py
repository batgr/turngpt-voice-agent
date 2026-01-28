# import os

import torch

# from pyannote.audio import Pipeline
from silero_vad import load_silero_vad

from audio_recorder import AudioRecorder
from speech_to_text import SpeechToText


class VoiceActivityDetector:
    def __init__(self, fs=16000, silence_limit=20, threshold=0.1) -> None:
        self.model = load_silero_vad()
        self.sampling_rate = fs
        self.silence_duration = 0
        self.silence_limit = silence_limit
        self.threshold = threshold
        self.buffer = []

    def is_speaking(self, wave):
        prob = self.model(wave, self.sampling_rate).item()
        return prob >= self.threshold

    def add_to_buffer(self, wave):
        self.buffer.append(wave)

    def get_full_audio(self):
        if not self.buffer:
            return None
        return torch.cat(self.buffer, dim=1)

    def reset(self):
        self.buffer = []
        self.silence_duration = 0


if __name__ == "__main__":
    stt = SpeechToText()
    recorder = AudioRecorder(duration=0.032)  # 32ms
    vad = VoiceActivityDetector()
    recorder.rec()

    try:
        while True:
            chunk = recorder.q.get()

            if chunk.size > 0:
                wave = torch.from_numpy(chunk).view(1, -1)

                if vad.is_speaking(wave):
                    vad.add_to_buffer(wave)
                    vad.silence_duration = 0
                else:
                    vad.silence_duration += 1
                    if (
                        len(vad.buffer) > 0
                        # and vad.silence_duration >= vad.silence_limit
                    ):
                        print(stt.to_text(vad.get_full_audio()))
                        # if vad.silence_duration >= vad.threshold:
                        #     print(stt.to_text(vad.get_full_audio()))
                        vad.reset()

    except KeyboardInterrupt:
        print("Stop")
