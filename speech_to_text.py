import torch
import whisper
from numpy import False_


class SpeechToText:
    def __init__(self, model="base") -> None:
        self.model = whisper.load_model(model)

    def to_text(self, wave):
        r = self.model.transcribe(wave[0], fp16=False)
        return r["text"]
