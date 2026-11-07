"""When the user's turn is over: the words decide early, silence decides in the end"""

CHUNK_SECONDS = 0.032


class TurnTaking:
    def __init__(self, short_pause=0.25, long_pause=1.2, threshold=0.3, min_speech=0.2, barge_in=0.3):
        self.short_chunks = round(short_pause / CHUNK_SECONDS)  # a pause long enough to ask TurnGPT
        self.long_chunks = round(long_pause / CHUNK_SECONDS)  # answer anyway
        self.threshold = threshold  # P(turn shift) that ends the turn after a short pause
        self.min_speech_chunks = round(min_speech / CHUNK_SECONDS)  # shorter is a noise, Whisper would invent words
        self.barge_in_chunks = round(barge_in / CHUNK_SECONDS)  # continuous speech needed to cut the agent off
        self.reset()

    def reset(self):
        self.speech = 0
        self.silence = 0
        self.turn_shift = None  # P(turn shift) of the current pause, None until asked

    def update(self, is_speech, agent_speaking=False):
        """One chunk of audio -> what to do: "listen", "discard" the audio, "interrupt" the agent,
        "ask" for the turn-shift probability, or "answer" """
        if is_speech:
            self.speech += 1
            self.silence = 0
            self.turn_shift = None
            return "interrupt" if agent_speaking and self.speech >= self.barge_in_chunks else "listen"

        self.silence += 1
        if self.speech == 0:
            return "listen"
        if agent_speaking or (self.silence >= self.short_chunks and self.speech < self.min_speech_chunks):
            self.reset()
            return "discard"
        if self.silence < self.short_chunks:
            return "listen"
        if self.turn_shift is None:
            return "ask"
        return "answer" if self.turn_shift >= self.threshold or self.silence >= self.long_chunks else "listen"
