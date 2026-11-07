# Multi-party AI dialog

A local voice agent that decides when the user has finished speaking from their words, not only from silence.

```text
microphone -> Silero VAD -> Whisper -> TurnGPT -> LLM (Ollama) -> Kokoro
```

After a short pause, Whisper transcribes the turn and [TurnGPT](https://arxiv.org/abs/2010.10874) gives the probability of a turn shift after the last word. A likely end answers at once; otherwise the agent waits for a longer pause or more speech. Sustained speech cuts the agent off.

```bash
uv sync
ollama pull llama3.2:3b-instruct-q4_K_M
uv run python -m turn.train_turn_gpt    # 3,000 steps; no English TurnGPT checkpoint is published
uv run python main.py                   # use a headset
uv run pytest
```

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

- Not yet tested with a real microphone and loudspeakers.
- Switchboard is casual conversation, so complete service requests can score low (P = 0.05 on the booking above): those turns wait for the long pause.
- The audio queues up while Whisper and the LLM run, and the history keeps the agent's whole reply even when it was cut off.
