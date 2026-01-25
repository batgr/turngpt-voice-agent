import numpy as np
import sounddevice as sd


class AudioRecorder:
    def __init__(self, fs=16000, duration=10):
        self.duration = 10
        self.fs = fs
        sd.default.samplerate = self.fs
        sd.default.channels = 1

        self.recording = None

    def rec(self):
        print("Recording")
        self.recording = sd.rec(int(self.duration * self.fs))
        sd.wait()
        print("Finish")
        return self.recording


if __name__ == "__main__":
    Recorder = AudioRecorder()

    recording = Recorder.rec()

    sd.play(recording, Recorder.fs)
    status = sd.wait()
    print(status)
