#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$SCRIPT_DIR/.." && pwd)/models"
HF="https://huggingface.co"
COMFY="Comfy-Org/vae-text-encorder-for-flux-klein-4b"

mkdir -p "$OUT"

dl() {
  local url="$1" dest="$2"
  [[ -f "$dest" ]] && echo "exists: $(basename "$dest")" && return
  echo "downloading: $(basename "$dest")"
  # -C - resumes a partial download; --retry retries on transient errors
  curl -fL --progress-bar --retry 5 --retry-delay 3 --retry-connrefused -C - -o "$dest" "$url" \
    || { rm -f "$dest"; exit 1; }
}

dl "$HF/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q8_0.gguf"                                         "$OUT/flux-2-klein-4b-Q8_0.gguf"
dl "$HF/$COMFY/resolve/main/split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors"  "$OUT/qwen_3_4b_fp4_flux2.safetensors"
dl "$HF/$COMFY/resolve/main/split_files/vae/flux2-vae.safetensors"                      "$OUT/flux2-vae.safetensors"

echo "done → $OUT"
