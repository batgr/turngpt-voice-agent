import threading

import sounddevice as sd
from kokoro import KPipeline


class TextToSpeech:
    def __init__(self, state) -> None:
        self.pipeline = KPipeline(lang_code="b")
        self.state = state
        self.thread = None
        self.stop_event = threading.Event()

    def to_speech(self, text):
        self.stop()

        self.thread = threading.Thread(target=self.play, args=(text,), daemon=True)
        self.thread.start()

    def generate(self, text):
        self.state.agent_speaking = True
        generator = self.pipeline(text, voice="af_heart")
        return generator

    def play(self, text):
        generator = self.generate(text)

        for _, __, audio in generator:
            if self.stop_event.is_set() or self.state.user_speaking:
                sd.stop()
                break
            sd.play(audio)
            sd.wait()
        self.state.agent_speaking = False

    def stop(self):
        self.stop_event.set()
        self.stop_event.clear()
        sd.stop()
        self.state.agent_speaking = False
