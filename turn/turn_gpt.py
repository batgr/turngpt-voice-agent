"""TurnGPT (Ekstedt and Skantze, 2020): a GPT-2 fine-tuned on spoken dialog, where the probability of the turn-shift
token <ts> after the last word tells whether the speaker has finished"""

import re

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

BASE_MODEL = "microsoft/DialoGPT-small"
SPECIAL_TOKENS = ["<ts>", "<speaker1>", "<speaker2>"]
MAX_TOKENS = 256


def normalize(text):
    """Text as an ASR gives it: lower case, no punctuation, no [laughter] tags"""
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text.lower())
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def encode(tokenizer, turns, last_turn_over=False):
    """turns: strings of alternating speakers -> input_ids and speaker_ids, with <ts> after each finished turn"""
    ts, first, second = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    input_ids, speaker_ids = [], []
    for i, turn in enumerate(turns):
        ids = tokenizer.encode(" " + normalize(turn))
        if i < len(turns) - 1 or last_turn_over:
            ids.append(ts)
        input_ids += ids
        speaker_ids += [first if i % 2 == 0 else second] * len(ids)
    return input_ids, speaker_ids


def extended_model(name=BASE_MODEL):
    """The pretrained model and its tokenizer, with the three tokens of TurnGPT"""
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    model = GPT2LMHeadModel.from_pretrained(name, dtype=torch.float32)  # saved as float16
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


class TurnGPT:
    def __init__(self, path="checkpoints/turn_gpt", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
        self.device = device
        self.ts = self.tokenizer.convert_tokens_to_ids("<ts>")

    @torch.no_grad()
    def turn_shift_probability(self, turns):
        """turns: the dialog so far, the last one being the current speaker's, unfinished or not"""
        input_ids, speaker_ids = encode(self.tokenizer, turns)
        input_ids = torch.tensor([input_ids[-MAX_TOKENS:]], device=self.device)
        speaker_ids = torch.tensor([speaker_ids[-MAX_TOKENS:]], device=self.device)
        logits = self.model(input_ids, token_type_ids=speaker_ids).logits[0, -1]
        return logits.softmax(-1)[self.ts].item()
