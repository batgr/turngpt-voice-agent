# Multi-party AI dialog

A local voice agent that decides when the user has finished speaking from their words, not only from silence.

```text
microphone -> Silero VAD -> Whisper -> TurnGPT -> LLM (Ollama) -> Kokoro
```

**Use a headset.** On loudspeakers the agent hears itself and cuts itself off.

```bash
uv sync
ollama pull llama3.2:3b-instruct-q4_K_M
uv run python -m turn.train_turn_gpt    # 3,000 steps; no English TurnGPT checkpoint is published
uv run python main.py                   # with a headset
uv run python main.py --speakers        # no headset: the microphone is ignored while the agent talks, so it cannot be interrupted
uv run pytest
```

## How it works

1. **Listening.** The microphone gives 32 ms chunks of 16 kHz audio. Silero VAD marks each chunk as speech or not, and the speech is buffered.
2. **Short pause, 0.25 s.** Speech under 0.2 s is dropped as noise. Otherwise Whisper transcribes the whole buffer, and [TurnGPT](https://arxiv.org/abs/2010.10874) reads the dialog so far and gives the probability of a turn shift after the last word.
3. **Decision.** If that probability is at least 0.3, the agent answers at once. If not, it keeps listening: new speech extends the same turn, and a pause of 1.2 s ends it anyway.
4. **Answer.** The LLM (Ollama) replies from the whole dialog, and Kokoro speaks the reply sentence by sentence.
5. **Interruption.** 0.3 s of continuous speech while the agent talks stops it, and the new speech becomes the next turn. Shorter speech is dropped.

So the words decide early and silence decides in the end: a finished sentence gets a quick answer, and a pause in the middle of a sentence is not cut off.

Each step is logged with its time: `user:`, `agent:`, `P(turn shift)`, `speech started`, `agent interrupted` and the microphone level.

| Path | Role |
|---|---|
| `main.py` | The loop |
| `speech/` | Microphone, VAD, speech to text, text to speech |
| `turn/turn_gpt.py` | TurnGPT: GPT-2 with a turn-shift token and speaker embeddings |
| `turn/train_turn_gpt.py` | Fine-tuning on Switchboard transcripts |
| `turn/turn_taking.py` | The pause and threshold rules |
| `agent/` | The LLM and its system prompt |

## Results

TurnGPT here is a reimplementation of the recipe (GPT-2 with `<ts>`, speaker embeddings and a weighted loss), fine-tuned from DialoGPT-small on Switchboard transcripts for 3,000 steps.

On held-out conversations, one seed, at every token (turn shifts 4.75 %): AUC 0.897, average precision 0.38. The threshold of 0.3 gives precision 0.32 and recall 0.58.

The turn rules were run end to end on synthetic speech with the real VAD, Whisper, TurnGPT, Ollama and Kokoro, the sound output replaced:

| Input | Result |
|---|---|
| "Thank you very much." | P = 0.70, answered after a short pause |
| "What time do you close?" | P = 0.29, answered after the long pause |
| "I would like to book a table for", 0.6 s pause, "two people tomorrow at eight." | P = 0.01, not cut off, one turn |
| 75 ms of noise | Ignored |
| Speech while the agent talks | Agent cut off after 0.33 s |

## Limits

- Checked by ear on one real voice, without measurements: the results above are on synthetic speech.
- Switchboard is casual conversation, so complete service requests can score low (P = 0.04 on the booking above): those turns wait for the long pause.
- The audio queues up while Whisper and the LLM run, and the history keeps the agent's whole reply even when it was cut off.

## Data and license

The code is under the [MIT License](LICENSE). The TurnGPT weights are not published and not covered by it: `turn/train_turn_gpt.py` trains them from [`amrit-kumar-jain/switchboard`](https://huggingface.co/datasets/amrit-kumar-jain/switchboard), a copy of Switchboard transcripts with no stated license. The original corpus is distributed by the [LDC](https://catalog.ldc.upenn.edu/LDC97S62) under its own agreement, so check your rights before using the data. DialoGPT, Whisper, Silero VAD, Kokoro and the LLM keep their own licenses.
