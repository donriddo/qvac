# @qvac/bci-whispercpp

Brain-Computer Interface (BCI) neural signal transcription adapter for qvac, built on top of [@qvac/transcription-whispercpp](../qvac-lib-infer-whispercpp).

Transcribes multi-channel neural signals (microelectrode array recordings) into English text using the [BrainWhisperer](https://github.com/Neuroprosthetics-Lab) model, achieving **8.86% Word Error Rate** — identical to the research notebook.

## Architecture

```
Neural Signal (.bin)
    │
    ▼
┌─────────────────────────────────────────┐
│  bci-whispercpp (thin adapter)          │
│                                         │
│  BCIWhispercpp.transcribe(signal.bin)   │
│       │                                 │
│       ▼                                 │
│  scripts/infer.py (Python backend)      │
│  ┌─────────────────────────────────┐    │
│  │ Gaussian smoothing (std=2, k=100)│   │
│  │ Day-specific projection          │   │
│  │ Conv1(512→384, k=7) + GELU      │   │
│  │ Conv2(384→384, k=3, s=2) + GELU │   │
│  │ 6-layer Transformer Encoder      │   │
│  │ LoRA-merged Whisper Decoder      │   │
│  │ Group beam search (4 beams)      │   │
│  └─────────────────────────────────┘    │
│       │                                 │
│       ▼                                 │
│  Transcribed text                       │
└─────────────────────────────────────────┘
```

The package delegates to `@qvac/transcription-whispercpp` for the underlying whisper.cpp engine. The Python inference backend (`scripts/infer.py`) runs the exact BrainWhisperer model with identical beam search parameters to guarantee notebook-matching output.

## Neural Signal Format

Binary files: `[uint32 numTimesteps, uint32 numChannels, float32[T*C] data]`

Each timestep = 20ms bin of neural activity. Channels = electrodes (typically 512).

## Usage

```javascript
const { BCIWhispercpp, computeWER } = require('@qvac/bci-whispercpp')

const bci = new BCIWhispercpp({
  checkpoint: '/path/to/epoch=93-val_wer=0.0910.ckpt',
  rnnArgs:    '/path/to/rnn_args.yaml',
  modelDir:   '/path/to/brainwhisperer-qvac',
  dataPath:   '/path/to/cleaned_val_data.pkl'  // for batch mode
})

// Single file
const result = bci.transcribe('signal.bin')
console.log(result.text)  // "Not too controversial."

// Batch (exact notebook match)
const results = bci.transcribeBatch()
for (const r of results) {
  console.log(`${r.text} (WER: ${(r.wer * 100).toFixed(1)}%)`)
}

// WER utility
const wer = computeWER('predicted text', 'reference text')
```

## Example

```bash
# Single file
node examples/transcribe-neural.js test/fixtures/neural_sample_0.bin

# Batch (all 5 test samples, exact notebook match)
node examples/transcribe-neural.js --batch
```

## Testing

```bash
node test/integration/bci-addon.test.js
```

## Prerequisites

- Python 3.10+ with: `torch`, `transformers`, `peft`, `lightning`, `omegaconf`, `scipy`
- The BrainWhisperer model files (checkpoint, rnn_args.yaml, model code)
- Neural signal test fixtures in `test/fixtures/`

## Model Conversion

To convert the BrainWhisperer checkpoint to GGML format (for future whisper.cpp native inference):

```bash
python3 scripts/convert-model.py \
  --checkpoint /path/to/epoch=93-val_wer=0.0910.ckpt \
  --output models/ggml-bci.bin
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| macOS arm64 | Tested | Full support |
| macOS x64 | Expected | Same Python backend |
| Linux x64 | Expected | Same Python backend |
| Windows | Expected | Python must be in PATH |

## License

Apache-2.0
