"""Fine-tunes TurnGPT on Switchboard transcripts. No English checkpoint is published, so this one is trained here.
uv run python -m turn.train_turn_gpt --steps 3000"""

import argparse
import random

import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from turn.turn_gpt import MAX_TOKENS, encode, extended_model, normalize

DATASET = "amrit-kumar-jain/switchboard"
TS_WEIGHT = 2.0  # against 1 for the other tokens, the ratio of TurnGPT's weight_eos_token / weight_regular_token


def dialog_turns(conversation):
    """Consecutive messages of a speaker merged, empty ones (only a [laughter] tag) dropped"""
    turns, speaker = [], None
    for message in conversation:
        text = normalize(message["content"])
        if not text:
            continue
        if message["role"] == speaker:
            turns[-1] += " " + text
        else:
            turns.append(text)
            speaker = message["role"]
    return turns


def windows(conversations, tokenizer):
    """Dialogs cut into windows of MAX_TOKENS"""
    out = []
    for conversation in conversations:
        turns = dialog_turns(conversation)
        if len(turns) < 2:
            continue
        input_ids, speaker_ids = encode(tokenizer, turns, last_turn_over=True)
        for start in range(0, len(input_ids) - 1, MAX_TOKENS):
            ids, speakers = input_ids[start : start + MAX_TOKENS], speaker_ids[start : start + MAX_TOKENS]
            if len(ids) > 8:
                out.append((ids, speakers))
    return out


def batch_of(rows, pad, device):
    length = max(len(ids) for ids, _ in rows)
    input_ids = torch.tensor([ids + [pad] * (length - len(ids)) for ids, _ in rows], device=device)
    speakers = torch.tensor([s + [s[-1]] * (length - len(s)) for _, s in rows], device=device)
    labels = torch.tensor([ids + [-100] * (length - len(ids)) for ids, _ in rows], device=device)
    return input_ids, speakers, labels


def loss_of(model, batch, weights):
    input_ids, speakers, labels = batch
    logits = model(input_ids, token_type_ids=speakers).logits
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), weight=weights, ignore_index=-100)


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, batch_size=8):
    """P(<ts>) at every position of the held-out dialogs against whether the turn really ends there"""
    ts = tokenizer.convert_tokens_to_ids("<ts>")
    probabilities, targets = [], []
    for i in range(0, len(rows), batch_size):
        input_ids, speakers, labels = batch_of(rows[i : i + batch_size], tokenizer.pad_token_id, device)
        p = model(input_ids, token_type_ids=speakers).logits[:, :-1].softmax(-1)[..., ts]
        kept = labels[:, 1:] != -100
        probabilities.append(p[kept].float().cpu())
        targets.append((labels[:, 1:][kept] == ts).cpu())
    probabilities, targets = torch.cat(probabilities).numpy(), torch.cat(targets).numpy()
    return {"auc": roc_auc_score(targets, probabilities), "average_precision": average_precision_score(targets, probabilities),
            "turn_shift_rate": float(targets.mean())}


def main(args):
    torch.manual_seed(0)
    random.seed(0)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = extended_model()
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    data = list(load_dataset(DATASET, split="train")["conversations"])
    random.shuffle(data)
    n_val = len(data) // 10  # held out by whole conversation
    val, train = windows(data[:n_val], tokenizer), windows(data[n_val:], tokenizer)
    print(f"{len(train)} training and {len(val)} validation windows", flush=True)
    print("before training", evaluate(model.eval(), val[:args.eval_windows], tokenizer, device), flush=True)

    weights = torch.ones(len(tokenizer), device=device)
    weights[tokenizer.convert_tokens_to_ids("<ts>")] = TS_WEIGHT
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(1, args.steps + 1):
        loss = loss_of(model, batch_of(random.sample(train, args.batch_size), tokenizer.pad_token_id, device), weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            print(step, evaluate(model.eval(), val[:args.eval_windows], tokenizer, device), flush=True)
            model.train()
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6.25e-5)  # TurnGPT's learning rate
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-windows", type=int, default=500)
    parser.add_argument("--output", default="checkpoints/turn_gpt")
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
