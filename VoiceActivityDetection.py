import os

import torch
from pyannote.audio import Pipeline


class VoiceActivityDetector:
    def __init__(self,src='pyannote/speaker-diarization-community-1') -> None:
        self.pipeline = Pipeline.from_pretrained(
            src,
            token=os.getenv("HUGGINGFACE_TOKEN"),
        )

    def output(self, input, fs):
        input = torch.from_numpy(input).view(1, -1)
        waveform = {"waveform": input, "sample_rate": fs}
        assert self.pipeline is not None
        out = self.pipeline(waveform)
        return out
