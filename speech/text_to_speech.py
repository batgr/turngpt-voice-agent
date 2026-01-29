import sounddevice as sd
from kokoro import KPipeline


class TextToSpeech:
    def __init__(self) -> None:
        self.pipeline = KPipeline(lang_code="b")

    def to_speech(self, text):
        generator = self.pipeline(text, voice="af_heart")
        for _, __, audio in generator:
            print(audio)
            sd.play(audio)
            sd.wait()
