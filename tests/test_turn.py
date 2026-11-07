from turn.turn_gpt import encode, normalize
from turn.turn_taking import TurnTaking


class FakeTokenizer:
    def convert_tokens_to_ids(self, tokens):
        return [100, 101, 102]

    def encode(self, text):
        return [len(word) for word in text.split()]


def test_normalize_gives_asr_like_text():
    assert normalize("Well, [laughter] I don't KNOW... maybe?") == "well i don't know maybe"


def test_only_finished_turns_end_with_a_turn_shift():
    input_ids, speaker_ids = encode(FakeTokenizer(), ["hi there", "hello"])
    assert input_ids == [2, 5, 100, 5]
    assert speaker_ids == [101, 101, 101, 102]
    assert encode(FakeTokenizer(), ["hi"], last_turn_over=True)[0] == [2, 100]


def speak(turns, chunks=10, agent_speaking=False):
    return [turns.update(True, agent_speaking) for _ in range(chunks)]


def pause(turns, chunks, agent_speaking=False):
    return [turns.update(False, agent_speaking) for _ in range(chunks)]


def test_a_short_pause_is_enough_when_the_words_end_the_turn():
    turns = TurnTaking()
    speak(turns)
    assert pause(turns, turns.short_chunks)[-1] == "ask"
    turns.turn_shift = 0.9
    assert turns.update(False) == "answer"


def test_unfinished_words_wait_for_the_long_pause():
    turns = TurnTaking()
    speak(turns)
    pause(turns, turns.short_chunks)
    turns.turn_shift = 0.05
    steps = pause(turns, turns.long_chunks - turns.short_chunks)
    assert steps[-2] == "listen" and steps[-1] == "answer"


def test_new_speech_cancels_the_pause_and_asks_again():
    turns = TurnTaking()
    speak(turns)
    pause(turns, turns.short_chunks)
    turns.turn_shift = 0.05
    speak(turns)
    assert pause(turns, turns.short_chunks)[-1] == "ask"


def test_silence_without_speech_never_asks():
    assert set(pause(TurnTaking(), 100)) == {"listen"}


def test_a_short_noise_is_discarded():
    turns = TurnTaking()
    speak(turns, 2)
    assert pause(turns, turns.short_chunks)[-1] == "discard"
    assert turns.speech == 0


def test_sustained_speech_interrupts_the_agent():
    turns = TurnTaking()
    assert speak(turns, turns.barge_in_chunks, agent_speaking=True)[-1] == "interrupt"


def test_a_noise_while_the_agent_speaks_does_not_interrupt():
    turns = TurnTaking()
    assert "interrupt" not in speak(turns, turns.barge_in_chunks - 1, agent_speaking=True)
    assert pause(turns, 1, agent_speaking=True) == ["discard"]
