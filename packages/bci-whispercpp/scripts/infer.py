#!/usr/bin/env python3
"""
BCI neural signal inference using the exact BrainWhisperer model.
Produces identical output to the Jupyter notebook.

Modes:
  Single file:
    python3 infer.py --signal <signal.bin> --checkpoint <model.ckpt> --args <rnn_args.yaml>

  Batch (exact notebook match):
    python3 infer.py --batch --data <cleaned_val_data.pkl> --checkpoint <model.ckpt> --args <rnn_args.yaml> --samples 0,1,2,3,4
"""

import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import torch


def remove_punctuation(s):
    s = re.sub(r"[^a-zA-Z\- ']", "", s)
    s = s.replace("- ", " ").lower().replace("--", "").replace(" '", "'").strip()
    return " ".join([w for w in s.split() if w])


def compute_wer(hypothesis, reference):
    hyp = hypothesis.lower().strip().split()
    ref = reference.lower().strip().split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def load_signal(path):
    with open(path, "rb") as f:
        T, C = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(T * C * 4), dtype=np.float32).reshape(T, C)
    return data, T, C


def run_batch(args):
    """Process via DataLoader (exact notebook match)."""
    import pickle
    from functools import partial
    from dataset import BaseNeuralTextDataset, collate_fn_flexible
    from utils import rename_batch_keys
    from pl_wrapper import LightningModel
    from transformers import WhisperProcessor

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    model = LightningModel.load_from_checkpoint(
        args.checkpoint, card_args_path=args.args, map_location="cpu")
    model.eval()
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    sample_indices = [int(x) for x in args.samples.split(",")]
    bs = max(len(sample_indices), 8)

    val_dataset = BaseNeuralTextDataset(data, source_dataset="card")
    collate_fn = partial(rename_batch_keys, collate_fn=collate_fn_flexible)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cpu")
    results = []

    for batch in val_loader:
        x, x_len = model.transform_data(
            batch["neural_feats"].to(device),
            batch["neural_time_bins"].to(device),
            mode="val",
        )
        with torch.no_grad():
            generated_ids = model.model.generate(
                x, x_len,
                batch["day"].to(device),
                sbj_idx=torch.zeros(len(batch["source_dataset"]),
                                     dtype=torch.long).to(device),
                num_beams=4,
                num_beam_groups=2,
                diversity_penalty=0.25,
                length_penalty=0.14,
                repetition_penalty=1.16,
                no_repeat_ngram_size=0,
            )
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        sentences = batch.get("sentence", [None] * len(texts))
        for idx_in_batch, (text, expected) in enumerate(zip(texts, sentences)):
            global_idx = idx_in_batch
            if global_idx not in sample_indices:
                continue
            result = {"index": global_idx, "text": text, "text_clean": remove_punctuation(text)}
            if expected:
                result["expected"] = expected
                result["expected_clean"] = remove_punctuation(expected)
                result["wer"] = compute_wer(result["text_clean"], result["expected_clean"])
            results.append(result)
        break  # first batch only

    for r in results:
        print(json.dumps(r))


def run_single(args):
    """Process a single .bin file."""
    from pl_wrapper import LightningModel
    from transformers import WhisperProcessor

    signal_data, T, C = load_signal(args.signal)

    model = LightningModel.load_from_checkpoint(
        args.checkpoint, card_args_path=args.args, map_location="cpu")
    model.eval()
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    features = torch.tensor(signal_data, dtype=torch.float32).unsqueeze(0)
    n_steps = torch.tensor([T], dtype=torch.long)
    day_idx = torch.tensor([args.day_idx], dtype=torch.long)
    device = torch.device("cpu")

    x, x_len = model.transform_data(features.to(device), n_steps.to(device), mode="val")

    with torch.no_grad():
        generated_ids = model.model.generate(
            x, x_len, day_idx.to(device),
            sbj_idx=torch.zeros(1, dtype=torch.long).to(device),
            num_beams=4, num_beam_groups=2,
            diversity_penalty=0.25, length_penalty=0.14,
            repetition_penalty=1.16, no_repeat_ngram_size=0,
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    result = {"text": text, "text_clean": remove_punctuation(text)}
    if args.expected:
        result["expected"] = args.expected
        result["expected_clean"] = remove_punctuation(args.expected)
        result["wer"] = compute_wer(result["text_clean"], result["expected_clean"])

    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Batch mode (exact notebook)")
    parser.add_argument("--signal", help="Path to .bin neural signal (single mode)")
    parser.add_argument("--data", help="Path to pickle data (batch mode)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args", required=True, help="Path to rnn_args.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--expected", default=None)
    parser.add_argument("--day-idx", type=int, default=0)
    parser.add_argument("--samples", default="0,1,2,3,4")
    args = parser.parse_args()

    if args.model_dir:
        sys.path.insert(0, args.model_dir)

    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
