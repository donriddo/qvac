#!/usr/bin/env python3
"""
Patch a whisper.cpp GGML model for BCI neural signal input.

Modifies the model so that our embedder's 384-dim output can be fed via
whisper_set_mel() and pass through to the transformer layers:

1. Changes n_mels from 80 → 384 (embedder output dim)
2. Replaces encoder.conv1.weight with identity-like kernel
3. Replaces encoder.conv2.weight with identity-like kernel
4. Zeroes out conv biases

Usage:
    python3 scripts/patch-ggml-model.py models/ggml-model.bin models/ggml-bci-patched.bin
"""

import struct
import sys
import os
import numpy as np
from pathlib import Path


def patch_model(input_path, output_path):
    with open(input_path, "rb") as f:
        original_data = f.read()

    # Parse header
    off = 0
    magic = struct.unpack_from("i", original_data, off)[0]; off += 4
    assert magic == 0x67676d6c, f"Bad magic: 0x{magic:08x}"

    # Header: vocab_size, max_source_positions, d_model, encoder_heads,
    #         encoder_layers, max_length, d_model, decoder_heads,
    #         decoder_layers, n_mels, ftype
    header = list(struct.unpack_from("11i", original_data, off))
    off += 44

    vocab_size = header[0]
    d_model = header[2]
    n_mels_orig = header[9]
    ftype_model = header[10]  # 0=f32, 1=f16

    print(f"vocab_size={vocab_size}, d_model={d_model}, "
          f"n_mels={n_mels_orig}, ftype={ftype_model}")

    NEW_MELS = d_model  # 384

    # Mel filters
    filter_rows = struct.unpack_from("i", original_data, off)[0]; off += 4
    filter_cols = struct.unpack_from("i", original_data, off)[0]; off += 4
    filter_bytes = filter_rows * filter_cols * 4
    off += filter_bytes
    print(f"Mel filters: {filter_rows}x{filter_cols} ({filter_bytes} bytes)")

    # Tokenizer
    n_tokens = struct.unpack_from("i", original_data, off)[0]; off += 4
    for _ in range(n_tokens):
        tlen = struct.unpack_from("i", original_data, off)[0]; off += 4
        off += tlen

    print(f"Tokenizer: {n_tokens} tokens")

    # Now parse tensors
    tensors = []
    while off < len(original_data):
        tensor_start = off
        n_dims = struct.unpack_from("i", original_data, off)[0]; off += 4
        name_len = struct.unpack_from("i", original_data, off)[0]; off += 4
        ftype = struct.unpack_from("i", original_data, off)[0]; off += 4

        dims = []
        for _ in range(n_dims):
            d = struct.unpack_from("i", original_data, off)[0]; off += 4
            dims.append(d)

        name = original_data[off:off + name_len].decode("utf-8")
        off += name_len

        # data size: ftype 0 = f32 (4 bytes), ftype 1 = f16 (2 bytes)
        n_elements = 1
        for d in dims:
            n_elements *= d
        elem_size = 4 if ftype == 0 else 2
        data_bytes = n_elements * elem_size
        data_start = off

        tensors.append({
            "name": name,
            "n_dims": n_dims,
            "dims": dims,
            "ftype": ftype,
            "data_start": data_start,
            "data_bytes": data_bytes,
            "n_elements": n_elements,
        })

        off += data_bytes

    print(f"Found {len(tensors)} tensors")

    # Build output file
    out = bytearray()

    # Magic
    out += struct.pack("i", 0x67676d6c)

    # Header with patched n_mels
    header[9] = NEW_MELS
    out += struct.pack("11i", *header)
    print(f"Patched n_mels: {n_mels_orig} → {NEW_MELS}")

    # Mel filters (write dummy for new size)
    new_filter_rows = NEW_MELS
    new_filter_cols = filter_cols
    out += struct.pack("i", new_filter_rows)
    out += struct.pack("i", new_filter_cols)
    out += np.zeros(new_filter_rows * new_filter_cols, dtype=np.float32).tobytes()
    print(f"Mel filters: {new_filter_rows}x{new_filter_cols} (zeroed)")

    # Tokenizer (copy verbatim)
    tok_start = 4 + 44 + 8 + filter_bytes
    tok_end = tok_start + 4  # n_tokens int
    n_tok_off = tok_start
    n_tok = struct.unpack_from("i", original_data, n_tok_off)[0]
    tok_cursor = n_tok_off + 4
    for _ in range(n_tok):
        tl = struct.unpack_from("i", original_data, tok_cursor)[0]
        tok_cursor += 4 + tl
    out += original_data[tok_start:tok_cursor]

    # Tensors - copy all, patch conv1 and conv2
    for t in tensors:
        name = t["name"]
        n_dims = t["n_dims"]
        dims = t["dims"]
        ftype = t["ftype"]
        n_elements = t["n_elements"]
        orig_data = original_data[t["data_start"]:t["data_start"] + t["data_bytes"]]

        if name == "encoder.conv1.weight":
            # Original dims in GGML: [3, n_mels_orig, d_model] reversed from PyTorch
            # which is [d_model, n_mels, kernel_size] → stored as [kernel_size, n_mels, d_model]
            # We need [3, NEW_MELS, d_model] with identity at center
            new_dims = [3, NEW_MELS, d_model]
            new_data = np.zeros((3, NEW_MELS, d_model), dtype=np.float16 if ftype == 1 else np.float32)
            new_data[1, :min(NEW_MELS, d_model), :min(NEW_MELS, d_model)] = np.eye(
                min(NEW_MELS, d_model), dtype=new_data.dtype)
            elem_size = 2 if ftype == 1 else 4
            raw = new_data.tobytes()

            # dims in GGML are stored as [kernel, n_mels, d_model]
            ggml_dims = [3, NEW_MELS, d_model]
            out += struct.pack("iii", n_dims, len(name.encode()), ftype)
            for d in ggml_dims:
                out += struct.pack("i", d)
            out += name.encode()
            out += raw
            print(f"  Patched {name}: {dims} → {ggml_dims} (identity)")
            continue

        elif name == "encoder.conv1.bias":
            # Zero the bias, keep shape
            new_data = np.zeros(n_elements, dtype=np.float32)
            out += struct.pack("iii", n_dims, len(name.encode()), 0)  # force f32
            for d in dims:
                out += struct.pack("i", d)
            out += name.encode()
            out += new_data.tobytes()
            print(f"  Patched {name}: zeros")
            continue

        elif name == "encoder.conv2.weight":
            # Identity conv2: [3, d_model, d_model]
            new_data = np.zeros((3, d_model, d_model), dtype=np.float16 if ftype == 1 else np.float32)
            new_data[1, :, :] = np.eye(d_model, dtype=new_data.dtype)
            raw = new_data.tobytes()

            out += struct.pack("iii", n_dims, len(name.encode()), ftype)
            for d in dims:
                out += struct.pack("i", d)
            out += name.encode()
            out += raw
            print(f"  Patched {name}: identity")
            continue

        elif name == "encoder.conv2.bias":
            new_data = np.zeros(n_elements, dtype=np.float32)
            out += struct.pack("iii", n_dims, len(name.encode()), 0)
            for d in dims:
                out += struct.pack("i", d)
            out += name.encode()
            out += new_data.tobytes()
            print(f"  Patched {name}: zeros")
            continue

        # Copy unchanged tensor
        out += struct.pack("iii", n_dims, len(name.encode()), ftype)
        for d in dims:
            out += struct.pack("i", d)
        out += name.encode()
        out += orig_data

    with open(output_path, "wb") as f:
        f.write(out)

    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 patch-ggml-model.py <input.bin> <output.bin>")
        sys.exit(1)
    patch_model(sys.argv[1], sys.argv[2])
