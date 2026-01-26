import queue

import numpy as np
import sounddevice as sd


class AudioRecorder:
    def __init__(self, fs=16000, duration=30):
        self.fs = fs
        self.block_size = int(fs * duration)
        self.q = queue.Queue()
        sd.default.samplerate = self.fs
        sd.default.channels = 1
        # self.recording = None

    def rec(self):
        print("recording...")
        self.stream = sd.InputStream(blocksize=self.block_size, callback=self.callback)
        self.stream.start()

    def callback(self, indata, frames, time, status):
        self.q.put(indata.copy())


if __name__ == "__main__":
    recorder = AudioRecorder(duration=5)
    recorder.rec()
    try:
        while True:
            chunk = recorder.q.get()

            volume = np.linalg.norm(chunk)
            print(f"Morceau reçu ! Volume : {volume:.4f}")

    except KeyboardInterrupt:
        print("stop recording")

    # recording = Recorder.rec()

    # sd.play(recording, Recorder.fs)
    # status = sd.wait()
    # print(status)
