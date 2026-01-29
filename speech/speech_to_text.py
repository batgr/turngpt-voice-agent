import whisper


class SpeechToText:
    def __init__(self, model="base") -> None:
        self.model = whisper.load_model(model)

    def to_text(self, wave):
        r = self.model.transcribe(wave.squeeze(), fp16=False, language="en")
        return r["text"]
