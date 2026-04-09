#!/usr/bin/env python3
"""
ONNX-accelerated BCI inference. Uses PyTorch model for preprocessing
(exact match with training pipeline) and ONNX Runtime for fast inference.

Usage:
  python3 onnx-infer.py --signal <neural.bin> --models-dir <onnx-dir> \
    --checkpoint <model.ckpt> --args <rnn_args.yaml> --model-dir <brainwhisperer-dir> \
    [--day-idx 1]

Output: JSON with { "text": "..." }
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import onnxruntime as ort


def load_signal(path):
    with open(path, "rb") as f:
        T, C = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(T * C * 4), dtype=np.float32).reshape(T, C)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0), T


def apply_day_projection(model, smoothed, day_idx_val):
    """Apply day projection from the loaded model (exact match)."""
    embedder = model.model.embedders[0]
    with torch.no_grad():
        if hasattr(embedder, 'day_As'):
            day_A = embedder.day_As[day_idx_val]
            day_B = embedder.day_Bs[day_idx_val]
            day_delta = day_A @ day_B
        elif hasattr(embedder, 'day_weights'):
            day_delta = embedder.day_weights[day_idx_val]
        else:
            return smoothed

        day_bias = embedder.day_biases[day_idx_val]

        day_number = embedder.sessions_to_days(torch.tensor(day_idx_val))
        month_idx = embedder.days_to_months(day_number)

        if hasattr(embedder, 'month_weights') and month_idx < len(embedder.month_weights):
            month_w = embedder.month_weights[month_idx]
            month_b = embedder.month_biases[month_idx]
            if month_w is not None:
                W = day_delta + month_w
                bias = day_bias + month_b
            else:
                W = day_delta
                bias = day_bias
        else:
            W = day_delta
            bias = day_bias

        x = torch.einsum("btd,dk->btk", smoothed, W) + bias.unsqueeze(0)
        x = embedder.day_layer_activation(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--day-idx", type=int, default=1)
    args = parser.parse_args()

    if args.model_dir:
        sys.path.insert(0, args.model_dir)

    from pl_wrapper import LightningModel

    pl_model = LightningModel.load_from_checkpoint(
        args.checkpoint, card_args_path=args.args, map_location="cpu")
    pl_model.eval()

    features, T = load_signal(args.signal)
    n_steps = torch.tensor([T], dtype=torch.long)

    x, x_len = pl_model.transform_data(features, n_steps, mode="val")
    projected = apply_day_projection(pl_model, x, args.day_idx)

    enc_path = os.path.join(args.models_dir, "bci_encoder.onnx")
    dec_path = os.path.join(args.models_dir, "bci_decoder.onnx")
    vocab_path = os.path.join(args.models_dir, "vocab.json")

    enc_sess = ort.InferenceSession(enc_path)
    dec_sess = ort.InferenceSession(dec_path)
    with open(vocab_path) as f:
        vocab = json.load(f)

    enc_out = enc_sess.run(None, {"projected_features": projected.numpy()})[0]

    input_ids = [50257, 50259, 50358, 50362]  # SOT, EN, TRANSCRIBE, NOTIMESTAMPS
    for _ in range(128):
        ids_np = np.array([input_ids], dtype=np.int64)
        logits = dec_sess.run(None, {
            "input_ids": ids_np,
            "encoder_hidden_states": enc_out,
        })[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token == 50256:  # EOT
            break
        input_ids.append(next_token)

    decoded = [t for t in input_ids[4:] if t < 50257]
    text = "".join(vocab.get(str(t), "") for t in decoded).strip()

    print(json.dumps({"text": text, "tokens": decoded}))


if __name__ == "__main__":
    main()
