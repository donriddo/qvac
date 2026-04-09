# @qvac/bci-whispercpp

Brain-Computer Interface (BCI) neural signal transcription addon for qvac, powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

This package adapts the whisper.cpp inference engine to accept multi-channel neural signals (e.g., from microelectrode arrays) instead of audio, and produces text transcriptions. It mirrors the JS API surface of `@qvac/transcription-whispercpp` but replaces audio input with neural signal input.

## Architecture

```
Neural Signals (multi-channel float arrays)
    │
    ▼
┌─────────────────────────┐
│   NeuralProcessor (C++) │  ← Gaussian smoothing, channel projection
│   - Smooth per channel  │
│   - Project to 1D       │
│   - Resample to 16kHz   │
└────────────┬────────────┘
             │  audio-like waveform
             ▼
┌─────────────────────────┐
│   whisper.cpp (vcpkg)   │  ← Unmodified whisper.cpp backend
│   - Mel spectrogram     │
│   - Encoder             │
│   - Decoder             │
└────────────┬────────────┘
             │
             ▼
        Text output
```

The neural signal processing pipeline:
1. **Gaussian smoothing** — reduces noise in neural firing rate estimates (per-channel 1D convolution with a Gaussian kernel, matching the BrainWhisperer preprocessing)
2. **Channel projection** — averages across all neural channels to produce a single-channel waveform
3. **Resampling** — upsamples from neural time resolution (50 Hz, 20ms bins) to audio sample rate (16kHz) via linear interpolation
4. **Normalization** — scales output to [-0.3, 0.3] amplitude range

## Neural Signal Format

Binary files with the following layout:

| Offset | Type    | Description          |
|--------|---------|----------------------|
| 0      | uint32  | Number of timesteps  |
| 4      | uint32  | Number of channels   |
| 8      | float32[] | Feature data (row-major: `features[t * channels + c]`) |

Each timestep represents a 20ms bin of neural activity. Channels correspond to individual electrodes in a microelectrode array (e.g., 256 or 512 channels).

## Installation

```bash
cd packages/bci-whispercpp
npm install
npm run build
```

### Prerequisites

- **Bare runtime** >= 1.19.0
- **CMake** >= 3.25
- **vcpkg** (configured via `vcpkg-configuration.json`)
- A whisper.cpp GGML model file (e.g., `ggml-tiny.en.bin`)

### Download Models

```bash
./scripts/download-models.sh
```

## Usage

### Low-level API (BCIInterface)

```javascript
const { BCIInterface } = require('@qvac/bci-whispercpp/bci')
const binding = require('@qvac/bci-whispercpp/binding')

const config = {
  contextParams: { model: '/path/to/ggml-tiny.en.bin' },
  whisperConfig: { language: 'en', temperature: 0.0 },
  miscConfig: { caption_enabled: false }
}

const onOutput = (addon, event, jobId, data, error) => {
  if (event === 'Output') console.log('Segment:', data.text)
  if (event === 'JobEnded') console.log('Done:', data)
  if (event === 'Error') console.error('Error:', error)
}

const model = new BCIInterface(binding, config, onOutput)
await model.activate()

// Batch mode
const neuralData = fs.readFileSync('signal.bin')
await model.runJob({ input: new Uint8Array(neuralData) })

// Streaming mode
await model.append({ type: 'neural', input: chunk1 })
await model.append({ type: 'neural', input: chunk2 })
await model.append({ type: 'end of job' })

await model.destroyInstance()
```

### High-level API (BCIWhispercpp)

```javascript
const { BCIWhispercpp, computeWER } = require('@qvac/bci-whispercpp')

const bci = new BCIWhispercpp(
  { modelPath: '/path/to/ggml-tiny.en.bin' },
  { whisperConfig: { language: 'en' } }
)

await bci.load()

// Transcribe a file
const result = await bci.transcribeFile('signal.bin')
console.log(result.text)

// Compute WER
const wer = computeWER(result.text, 'expected transcription')
console.log(`WER: ${(wer * 100).toFixed(1)}%`)

await bci.destroy()
```

### Example Script

```bash
bare examples/transcribe-neural.js test/fixtures/neural_sample_0.bin models/ggml-tiny.en.bin
```

## Testing

### Integration Tests

```bash
WHISPER_MODEL_PATH=models/ggml-tiny.en.bin npm run test:integration
```

### C++ Unit Tests

```bash
npm run test:cpp
```

## Configuration

### whisperConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `"en"` | Language code |
| `n_threads` | number | `0` (auto) | Number of threads |
| `temperature` | number | `0.0` | Sampling temperature |
| `suppress_nst` | boolean | `true` | Suppress non-speech tokens |
| `duration_ms` | number | `0` | Max duration in ms (0 = unlimited) |

### bciConfig (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smooth_kernel_std` | number | `2.0` | Gaussian smoothing kernel std |
| `smooth_kernel_size` | number | `20` | Smoothing kernel size |
| `sample_rate` | number | `16000` | Target sample rate for whisper.cpp |

### contextParams

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | **Required.** Path to GGML model file |
| `use_gpu` | boolean | Enable GPU acceleration |
| `flash_attn` | boolean | Enable flash attention |
| `gpu_device` | number | GPU device index |

## Platform Support

### Verified

| Platform | Architecture | Status |
|----------|-------------|--------|
| macOS (Darwin) | arm64 (Apple Silicon) | ✅ Tested |

### Feasibility Assessment

| Platform | Architecture | Feasibility | Notes |
|----------|-------------|-------------|-------|
| macOS | x86_64 | ✅ High | Same build system, minor toolchain changes |
| Linux | x64 | ✅ High | Whisper.cpp has full Linux support; build with `libc++` |
| Linux | arm64 | ✅ High | Cross-compile via vcpkg triplets (same as transcription-whispercpp) |
| Windows | x64 | ✅ High | Whisper.cpp supports MSVC; add `msvcrt.lib` link (already in CMake) |
| Android | arm64 | 🟡 Medium | Requires NDK toolchain; transcription-whispercpp already supports this |
| iOS | arm64 | 🟡 Medium | Requires Xcode toolchain; transcription-whispercpp has iOS prebuilds |

The build system (CMake + vcpkg + bare-make) is the same as `@qvac/transcription-whispercpp`, which already supports all these platforms. Porting primarily requires:
1. Adding platform-specific vcpkg triplets (can copy from transcription-whispercpp)
2. Setting up CI matrix entries for each platform
3. Testing neural signal I/O on each target

## Limitations

- **Standard whisper.cpp model**: The current implementation uses a standard Whisper model (e.g., `whisper-tiny.en`). For accurate neural-to-text decoding, a BCI-trained model (like the BrainWhisperer model with LoRA-adapted decoder) must be converted to GGML format.
- **Signal projection**: The channel-averaging projection is a simplified stand-in for the learned neural embedder from the BrainWhisperer architecture. Production use requires exporting the trained embedding weights.
- **No LoRA support in whisper.cpp**: The BrainWhisperer model uses LoRA adapters on the Whisper decoder. Supporting this requires either (a) merging LoRA weights into the base model before GGML conversion, or (b) adding LoRA inference support to whisper.cpp.

## License

Apache-2.0
