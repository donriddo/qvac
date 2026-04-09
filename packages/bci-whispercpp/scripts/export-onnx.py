#!/usr/bin/env python3
"""
Export BrainWhisperer encoder and decoder to ONNX for C++ inference.

Usage:
  python3 scripts/export-onnx.py \
    --checkpoint /path/to/epoch=93-val_wer=0.0910.ckpt \
    --args /path/to/rnn_args.yaml \
    --model-dir /path/to/brainwhisperer-qvac \
    --output-dir models/onnx

Produces:
  - bci_encoder.onnx: projected_features[1,T,512] → encoder_out[1,1500,384]
    (Takes day-projected + smoothed features; conv1/conv2/pos_enc/transformer inside)
  - bci_decoder.onnx: input_ids[1,S] + encoder_out[1,1500,384] → logits[1,S,51864]
  - bci_config.json: tokenizer IDs and decode params
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderWrapper(nn.Module):
    """Wraps conv layers + positional encoding + transformer encoder for ONNX export.

    Input: day-projected features [1, T, 512] (after Gaussian smoothing + day projection)
    Output: encoder hidden states [1, 1500, 384]

    Day projection is done outside ONNX (in C++) because SessionsToDays
    uses data-dependent indexing that can't be traced.
    """

    def __init__(self, brainwhisperer):
        super().__init__()
        embedder = brainwhisperer.embedders[0]
        self.conv1 = embedder.conv1
        self.conv2 = embedder.conv2
        self.max_source_positions = embedder.max_source_positions
        self.stride_2 = embedder.conv2.stride[0]

        # Bake the day encoding for day_idx=1 (session index 1) into the model
        # This avoids the SessionsToDays lookup at runtime
        with torch.no_grad():
            day_number = embedder.sessions_to_days(torch.tensor(1))
            de = embedder.de(day_number)
            if de.dim() == 2:
                de = de.unsqueeze(1)
        self.register_buffer("day_encoding", de)
        self.embed_dim = brainwhisperer.whisper.config.d_model

        self.encoder = brainwhisperer.whisper.model.encoder

    def forward(self, projected_features):
        # projected_features: [batch, T, 512] - already smoothed and day-projected
        x = projected_features.permute(0, 2, 1)  # [batch, 512, T]

        expected_len = self.max_source_positions * self.stride_2
        pad_size = expected_len - x.shape[-1]
        if pad_size > 0:
            x = F.pad(x, (0, pad_size), mode="constant", value=0)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        inputs_embeds = x.permute(0, 2, 1)  # [batch, 1500, 384]

        # Add day encoding (goes into second half of dims)
        padded_de = torch.zeros(
            1, 1, inputs_embeds.shape[-1], device=inputs_embeds.device
        )
        padded_de[..., -self.day_encoding.shape[-1]:] = self.day_encoding
        inputs_embeds = inputs_embeds + padded_de

        # Feed to encoder (permute back for encoder format: [batch, d_model, seq_len])
        encoder_out = self.encoder(inputs_embeds.permute(0, 2, 1))
        return encoder_out.last_hidden_state


class DecoderWrapper(nn.Module):
    """Wraps decoder + proj_out for ONNX export (no KV cache for simplicity)."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.whisper.model.decoder
        self.proj_out = model.whisper.proj_out

    def forward(self, input_ids, encoder_hidden_states):
        decoder_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        )
        logits = self.proj_out(decoder_out.last_hidden_state)
        return logits


def load_model(args):
    if args.model_dir:
        sys.path.insert(0, args.model_dir)

    from pl_wrapper import LightningModel

    model = LightningModel.load_from_checkpoint(
        args.checkpoint, card_args_path=args.args, map_location="cpu"
    )
    model.eval()
    return model


def gauss_smooth(data, kernel_std=2.0, kernel_size=100):
    """Matches pl_wrapper.LightningModel.gauss_smooth"""
    kernel = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (kernel / kernel_std) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    n_channels = data.shape[-1]
    kernel = kernel.expand(n_channels, -1, -1)
    data_t = data.permute(0, 2, 1)
    pad = kernel_size // 2
    data_padded = torch.nn.functional.pad(data_t, (pad, pad - 1), mode="constant", value=0)
    smoothed = torch.nn.functional.conv1d(data_padded, kernel, groups=n_channels)
    return smoothed.permute(0, 2, 1)


def load_signal(path):
    with open(path, "rb") as f:
        T, C = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(T * C * 4), dtype=np.float32).reshape(T, C)
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0), T


def apply_day_projection_python(model, smoothed, day_idx_val):
    """Apply the day projection from the embedder (outside ONNX trace)."""
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

        # Month projection
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
        x = embedder.day_layer_activation(x)  # softsign
        return x


def export_encoder(model, args, output_dir):
    encoder_wrapper = EncoderWrapper(model.model)
    encoder_wrapper.eval()

    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures", "neural_sample_2.bin"
    )
    features, T = load_signal(sample_path)
    smoothed = gauss_smooth(features)
    projected = apply_day_projection_python(model, smoothed, day_idx_val=1)

    with torch.no_grad():
        pt_out = encoder_wrapper(projected)
    print(f"Encoder PyTorch output shape: {pt_out.shape}")
    print(f"  range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")

    onnx_path = os.path.join(output_dir, "bci_encoder.onnx")
    torch.onnx.export(
        encoder_wrapper,
        (projected,),
        onnx_path,
        input_names=["projected_features"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "projected_features": {1: "time"},
            "encoder_hidden_states": {1: "seq_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported encoder: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {
        "projected_features": projected.numpy(),
    })[0]
    diff = np.abs(pt_out.numpy() - onnx_out).max()
    print(f"  Max diff vs PyTorch: {diff:.7f}")
    return pt_out


def export_decoder(model, encoder_out, output_dir):
    decoder_wrapper = DecoderWrapper(model.model)
    decoder_wrapper.eval()

    input_ids = torch.tensor([[50257]], dtype=torch.long)

    with torch.no_grad():
        pt_logits = decoder_wrapper(input_ids, encoder_out)
    print(f"\nDecoder PyTorch logits shape: {pt_logits.shape}")

    onnx_path = os.path.join(output_dir, "bci_decoder.onnx")
    torch.onnx.export(
        decoder_wrapper,
        (input_ids, encoder_out),
        onnx_path,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported decoder: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    onnx_logits = sess.run(None, {
        "input_ids": input_ids.numpy(),
        "encoder_hidden_states": encoder_out.numpy(),
    })[0]
    diff = np.abs(pt_logits.numpy() - onnx_logits).max()
    print(f"  Max diff vs PyTorch: {diff:.7f}")


def verify_greedy_decode(model, output_dir):
    """Run greedy decode with ONNX models and compare to PyTorch beam search."""
    import onnxruntime as ort
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    tokenizer = processor.tokenizer

    enc_sess = ort.InferenceSession(os.path.join(output_dir, "bci_encoder.onnx"))
    dec_sess = ort.InferenceSession(os.path.join(output_dir, "bci_decoder.onnx"))

    fixtures_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "fixtures"
    )
    manifest = json.load(open(os.path.join(fixtures_dir, "manifest.json")))
    py_preds = json.load(open(os.path.join(fixtures_dir, "python_predictions.json")))

    print(f"\n{'='*60}")
    print("ONNX Greedy Decode Verification")
    print(f"{'='*60}")

    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    for i, sample in enumerate(manifest["samples"]):
        signal_path = os.path.join(fixtures_dir, sample["file"])
        features, T = load_signal(signal_path)
        smoothed = gauss_smooth(features)
        day_idx_val = sample.get("day_idx", 1)
        projected = apply_day_projection_python(model, smoothed, day_idx_val)

        # ONNX encoder
        enc_out = enc_sess.run(None, {
            "projected_features": projected.numpy(),
        })[0]

        # Greedy decode
        SOT = 50257
        EN = 50259
        TRANSCRIBE = 50358
        NOTIMESTAMPS = 50362
        EOT = 50256

        input_ids = [SOT, EN, TRANSCRIBE, NOTIMESTAMPS]
        max_tokens = 128

        for _ in range(max_tokens):
            ids_np = np.array([input_ids], dtype=np.int64)
            logits = dec_sess.run(None, {
                "input_ids": ids_np,
                "encoder_hidden_states": enc_out,
            })[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token == EOT:
                break
            input_ids.append(next_token)

        decoded_ids = [t for t in input_ids[4:] if t < 50257]
        onnx_text = tokenizer.decode(decoded_ids, skip_special_tokens=True).strip()

        # PyTorch beam search for comparison
        with torch.no_grad():
            x, x_len = model.transform_data(
                features, torch.tensor([T], dtype=torch.long), mode="val"
            )
            gen_ids = model.model.generate(
                x, x_len, torch.tensor([day_idx_val], dtype=torch.long),
                sbj_idx=torch.zeros(1, dtype=torch.long),
                num_beams=4, num_beam_groups=2,
                diversity_penalty=0.25, length_penalty=0.14,
                repetition_penalty=1.16,
            )
            beam_text = proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        py_pred = py_preds[i]["prediction"] if i < len(py_preds) else "N/A"

        print(f"\n  Sample {i}: {sample['file']}")
        print(f"    Expected:       \"{sample['expected_text']}\"")
        print(f"    Python beam:    \"{beam_text}\"")
        print(f"    Cached py pred: \"{py_pred}\"")
        print(f"    ONNX greedy:    \"{onnx_text}\"")


def save_config(model, output_dir):
    config = {
        "sot_token": 50257,
        "eot_token": 50256,
        "en_token": 50259,
        "transcribe_token": 50358,
        "notimestamps_token": 50362,
        "vocab_size": model.model.whisper.config.vocab_size,
        "d_model": model.model.whisper.config.d_model,
        "max_target_positions": model.model.whisper.config.max_target_positions,
        "max_source_positions": model.model.whisper.config.max_source_positions,
        "smooth_kernel_std": 2.0,
        "smooth_kernel_size": 100,
        "num_channels": 512,
    }
    path = os.path.join(output_dir, "bci_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--verify", action="store_true", help="Run greedy decode verification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args)

    encoder_out = export_encoder(model, args, args.output_dir)
    export_decoder(model, encoder_out, args.output_dir)
    save_config(model, args.output_dir)

    if args.verify:
        verify_greedy_decode(model, args.output_dir)


if __name__ == "__main__":
    main()
