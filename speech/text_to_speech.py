import threading
import time

import sounddevice as sd
from kokoro import KPipeline


class TextToSpeech:
    def __init__(self) -> None:
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        self.thread = None
        self.stop_event = threading.Event()

    @property
    def speaking(self):
        return self.thread is not None and self.thread.is_alive()

    def to_speech(self, text):
        self.stop()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.play, args=(text, self.stop_event), daemon=True)
        self.thread.start()

    def play(self, text, stop_event):
        for _, __, audio in self.pipeline(text, voice="af_heart"):
            if stop_event.is_set():
                return
            sd.play(audio.numpy(), 24000)
            while sd.get_stream().active:
                if stop_event.is_set():
                    sd.stop()
                    return
                time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
