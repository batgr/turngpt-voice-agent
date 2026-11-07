import whisper


class SpeechToText:
    def __init__(self, model="base.en") -> None:
        self.model = whisper.load_model(model)

    def to_text(self, wave):
        return self.model.transcribe(wave.squeeze().numpy(), fp16=False, language="en")["text"].strip()
