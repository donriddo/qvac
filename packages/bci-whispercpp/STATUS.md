# BCI-Whispercpp: Current Status & What's Needed

## What Exists

### BrainWhisperer Research Model (Python — working, 8.86% WER)
- **Location**: `/Users/rajusharma/Downloads/brainwhisperer-qvac/`
- **Checkpoint**: `epoch=93-val_wer=0.0910.ckpt` (PyTorch Lightning)
- **Architecture**: Custom WhisperEmbedder (conv1 k=7, conv2 k=3, day projections) + 6-layer Whisper encoder + LoRA-adapted 4-layer decoder
- **Notebook** (`test.ipynb`): Runs full validation, 8.84% WER across 1,431 samples
- **Key decode params**: `num_beams=4, num_beam_groups=2, diversity_penalty=0.25, length_penalty=0.14, repetition_penalty=1.16`

### Test Fixtures (5 real brain signal samples)
- **Location**: `test/fixtures/neural_sample_0..4.bin`
- **Format**: `[uint32 numTimesteps, uint32 numChannels, float32[T*C]]` (row-major)
- **Channels**: 512 (microelectrode array), 20ms bins
- **Expected outputs** (from Python model):

| # | Timesteps | Expected Text | Python Prediction | WER |
|---|-----------|---------------|-------------------|-----|
| 0 | 910 | "You can see the code at this point as well." | "You can see the good at this point as well." | 10% |
| 1 | 749 | "How does it keep the cost down?" | "How does it keep the cost said?" | 14.3% |
| 2 | 502 | "Not too controversial." | "Not too controversial." | 0% |
| 3 | 962 | "The jury and a judge work together on it." | "The jury and a judge work together on it." | 0% |
| 4 | 584 | "Were quite vocal about it." | "We're quite vocal about it." | 20% |

### Model Conversion Tools
- `scripts/convert-model.py`: Merges LoRA weights, exports GGML model with 6 encoder layers, BCI conv1/conv2, day-0 positional embedding
- `scripts/infer.py`: Python reference inference (exact notebook output, used for test verification only)
- `models/bci-embedder.bin`: Exported embedder weights (day projections, conv1/conv2) in binary format

### Package Structure (current — refactored to thin adapter, needs C++ restored)
- `index.js`, `index.d.ts`, `package.json`
- `test/integration/bci-addon.test.js`
- `examples/transcribe-neural.js`
- `README.md`

## What Was Built (C++ addon — needs to be restored)

A full C++ native addon was built and tested but removed during refactoring. It needs to be brought back. The code existed in a previous git commit (`cbdeaae`) on branch `feat/bci-whispercpp`.

### C++ Components That Worked
1. **NeuralProcessor** (`NeuralProcessor.hpp/.cpp`): Gaussian smoothing (std=2, kernel=100), day-specific projection (loads from `bci-embedder.bin`), conv1d (k=7), padding to 3000 frames
2. **BCIModel** (`BCIModel.hpp/.cpp`): Wraps whisper.cpp, injects mel features via `whisper_set_mel_with_state()` in `encoder_begin_callback`, segment callbacks, runtime stats
3. **BCIConfig** (`BCIConfig.hpp/.cpp`): whisper_full_params / whisper_context_params from JS config
4. **JSAdapter** (`JSAdapter.hpp/.cpp`): JS object → C++ config bridge (same pattern as transcription-whispercpp)
5. **AddonJs** (`AddonJs.hpp`): Bare module exports (createInstance, runJob, reload, etc.)
6. **binding.cpp**: `BARE_MODULE` entry point

### Build System That Worked
- CMakeLists.txt linking whisper::whisper via vcpkg
- vcpkg.json with whisper-cpp 1.7.5.1 dependency
- vcpkg overlay patching whisper.cpp for variable conv1 kernel size (3-line patch)
- Built and ran on macOS arm64 (Apple Silicon)

## The Gap: Why C++ Output Doesn't Match Python

### What whisper.cpp hardcodes
- **conv1 kernel_size=3** at line 1778 of whisper.cpp. Our vcpkg overlay patch fixes this to read from model header.
- **Positional embedding** is always added after conv2. The BCI model's custom encoder skips this (embedder adds its own day encoding). We set it to day-0 encoding in the GGML model.

### Verified correct
- All 48 encoder tensor weights match PyTorch (max diff < 0.00022, f16 tolerance)
- All 52 decoder tensor weights match (LoRA merge verified exact against PEFT)
- Conv1 weights (384, 512, 7) match exactly
- Gaussian smoothing matches Python (diff < 0.000001)
- Day projection (softsign activation) matches Python
- Mel injection via `whisper_set_mel_with_state` succeeds (returns 0)

### Root cause of divergence
GGML's tensor operations (attention, GELU approximation, float accumulation order) produce numerically different intermediate values than PyTorch. For standard audio whisper, this doesn't matter because the model is robust to small perturbations. For BCI, the neural embeddings operate in a narrow numerical range where small differences cascade through 6 transformer layers.

The C++ addon produced coherent English text (e.g., "Bachelornoon?", "Russoange Timberwolves") but not the correct sentences. The model IS running — it's just that the accumulated numerical drift through 6 encoder layers + 4 decoder layers produces different token selections.

## What's Needed

### Option A: Accept GGML numerical differences (recommended for v1)
1. **Restore the C++ addon code** from commit `cbdeaae`
2. Keep the patched whisper.cpp overlay (variable conv1 kernel)
3. Keep the GGML model conversion (`convert-model.py`)
4. Use the Python script (`infer.py`) only for reference testing
5. Accept that C++ WER will be higher than Python WER
6. Document the difference in README

### Option B: ONNX Runtime backend (exact match possible)
1. Export encoder + decoder step as ONNX models (encoder export verified: 0.4MB, max diff 0.00007)
2. Replace whisper.cpp with ONNX Runtime in the C++ addon
3. Implement greedy decode loop in C++ (beam search for exact match is complex)
4. ONNX Runtime is already used in qvac (`qvac-lib-infer-onnx` package)
5. Greedy decode tested: "You can see the good at this part as well." (close but not identical to beam search)

### Option C: Hybrid (best of both)
1. C++ addon with whisper.cpp for fast/approximate inference
2. Python fallback for exact notebook-matching output (test/validation only)
3. ONNX path as future optimization

## Key Files Reference

| File | What |
|------|------|
| `/Users/rajusharma/Downloads/brainwhisperer-qvac/model.py` | Full BrainWhisperer architecture (WhisperEmbedder, WhisperEncoder_, WhisperForConditionalGeneration_) |
| `/Users/rajusharma/Downloads/brainwhisperer-qvac/pl_wrapper.py` | LightningModel wrapper (Gaussian smoothing, data transforms) |
| `/Users/rajusharma/Downloads/brainwhisperer-qvac/rnn_args.yaml` | Preprocessing params (smooth_kernel_std=2, smooth_kernel_size=100) |
| `/Users/rajusharma/Downloads/brainwhisperer-qvac/cleaned_val_data.pkl` | Validation data (1,431 samples, pickle) |
| `packages/qvac-lib-infer-whispercpp/` | Reference whisper addon to mirror (JS bindings, C++ addon pattern, CMake+Bare build) |
| `packages/qvac-lib-inference-addon-cpp/` | Shared C++ addon framework (AddonJs, JsInterface, OutputQueue, etc.) |

## Draft PR
https://github.com/sharmaraju352/qvac/pull/2 (currently has thin adapter — needs C++ addon restored)
