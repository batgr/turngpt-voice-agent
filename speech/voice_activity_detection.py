import torch
from silero_vad import load_silero_vad


class VoiceActivityDetector:
    def __init__(self, fs=16000, threshold=0.5) -> None:
        self.model = load_silero_vad()
        self.sampling_rate = fs
        self.threshold = threshold
        self.buffer = []

    def is_speaking(self, wave):
        prob = self.model(wave, self.sampling_rate).item()
        return prob >= self.threshold

    def add_to_buffer(self, wave):
        self.buffer.append(wave)

    def get_full_audio(self):
        return torch.cat(self.buffer, dim=1)

    def reset(self):
        self.buffer = []
        self.model.reset_states()
