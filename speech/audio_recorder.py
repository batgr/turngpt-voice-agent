import queue

import sounddevice as sd


class AudioRecorder:
    def __init__(self, fs=16000, duration=0.032):
        self.block_size = int(fs * duration)
        self.fs = fs
        self.q = queue.Queue()

    def rec(self):
        self.stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            blocksize=self.block_size,
            callback=self.callback,
            dtype="float32",
        )
        self.stream.start()

    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def chunks(self):
        while True:
            yield self.q.get()
