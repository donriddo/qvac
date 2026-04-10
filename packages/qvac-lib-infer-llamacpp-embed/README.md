# qvac-lib-infer-llamacpp-embed

This native C++ addon, built using the `Bare` Runtime, simplifies running text embedding models to enable efficient generation of high-quality contextual text embeddings. It provides an easy interface to load, execute, and manage embedding model instances.

## Table of Contents

- [Supported platforms](#supported-platforms)
- [Installation](#installation)
- [Building from Source](#building-from-source)
- [Usage](#usage)
  - [1. Import the Model Class](#1-import-the-model-class)
  - [2. Create the `args` obj](#2-create-the-args-obj)
  - [3. Create `config`](#3-create-config)
  - [4. Instanstiate the model](#4-instanstiate-the-model)
  - [5. Load the model](#5-load-the-model)
  - [6. Generate embeddings for input sequence](#6-generate-embeddings-for-input-sequence)
  - [7. Release Resources](#7-release-resources)
- [API behavior by state](#api-behavior-by-state)
- [Quickstart Example](#quickstart-example)
- [Other Examples](#other-examples)
- [Benchmarking](#benchmarking)
- [Tests](#tests)
- [Glossary](#glossary)
- [License](#license)

## Supported platforms

| Platform | Architecture | Min Version | Status | GPU Support |
|----------|-------------|-------------|--------|-------------|
| macOS | arm64, x64 | 14.0+ | ✅ Tier 1 | Metal |
| iOS | arm64 | 17.0+ | ✅ Tier 1 | Metal |
| Linux | arm64, x64 | Ubuntu-22+ | ✅ Tier 1 | Vulkan |
| Android | arm64 | 12+ | ✅ Tier 1 | Vulkan, OpenCL (Adreno 700+) |
| Windows | x64 | 10+ | ✅ Tier 1 | Vulkan |

**Dependencies:**
- qvac-lib-inference-addon-cpp (≥1.1.2): C++ addon framework
- qvac-fabric-llm.cpp (≥7248.2.3): Inference engine
- Bare Runtime (≥1.24.0): JavaScript runtime
- Linux requires Clang/LLVM 19 with libc++


## Installation

### Prerequisites

Ensure that the `Bare` Runtime is installed globally on your system. If it's not already installed, you can install it using:

```bash
npm install -g bare@latest
```

### Installing the Package

```bash
npm install @qvac/embed-llamacpp@latest
```

## Building from Source

See [build.md](./build.md) for detailed instructions on how to build the addon from source.

## Usage

### 1. Import the Model Class

```js
const GGMLBert = require('@qvac/embed-llamacpp')
```

### 2. Create the `args` obj

```js
const path = require('bare-path')

const args = {
  files: { model: [path.join(dirPath, modelName)] },
  config: {
    device: 'gpu',
    gpu_layers: '99',
    batch_size: '1024',
    ctx_size: '512'
  },
  logger: console,
  opts: { stats: true }
}
```

The `args` obj contains the following properties:

* `files.model`: An array of absolute paths to the model file(s) on disk. For sharded models, provide all shard paths.
* `config`: A dictionary of hyper-parameters used to tweak the behaviour of the model (see [Create `config`](#3-create-config) below).
* `logger`: This property is used to create a [`QvacLogger`](../logging) instance, which handles all logging functionality.
* `opts.stats`: This flag determines whether to calculate inference stats.

#### Sharded model usage

The addon does not discover companion files on disk — the caller MUST pass every file the model needs, in order, via `files.model`. For sharded GGUF models this includes the `.tensors.txt` companion file followed by each `.gguf` shard in numerical order.

```js
const path = require('bare-path')

const dir = '/path/to/models'
const model = new GGMLBert({
  files: {
    model: [
      path.join(dir, 'gte-large.Q2_K.tensors.txt'),
      path.join(dir, 'gte-large.Q2_K-00001-of-00005.gguf'),
      path.join(dir, 'gte-large.Q2_K-00002-of-00005.gguf'),
      path.join(dir, 'gte-large.Q2_K-00003-of-00005.gguf'),
      path.join(dir, 'gte-large.Q2_K-00004-of-00005.gguf'),
      path.join(dir, 'gte-large.Q2_K-00005-of-00005.gguf')
    ]
  },
  config: { device: 'gpu', gpu_layers: '99' },
  logger: console,
  opts: { stats: true }
})
```

Rules for the `files.model` array:

* **Order matters.** The `.tensors.txt` file must come first, then shards in ascending numerical order (`00001-of-00005`, `00002-of-00005`, ...).
* **All shards are required.** Missing any shard or the `.tensors.txt` companion will fail loading.
* **Non-sharded models** pass a single absolute path: `files: { model: [modelPath] }`.
* **Absolute paths only.** The addon reads each file directly via `bare-fs` during `load()`.

### 3. Create `config`

The `config` is a dictionary (object) consisting of hyper-parameters which can be used to tweak the behaviour of the model.
All parameter values should be strings.

| Parameter         | Range / Type                                | Default                      | Description                                           |
|-------------------|---------------------------------------------|------------------------------|-------------------------------------------------------|
| -dev    | `"gpu"` or `"cpu"`                          | `"gpu"`                      | Device to run inference on                            |
| -ngl | integer                                    | 0            | Number of model layers to offload to GPU              |
|--batch-size  | integer                                     | 2048                         | Tokens for processing multiple prompts together             |
| --pooling         | `{none,mean,cls,last,rank}`                 | model default                | Pooling type for embeddings                           |
| --attention       | `{causal,non-causal}`                       | model default                | Attention type for embeddings                         |
| --embd-normalize  | integer                                     | 2                            | Embedding normalization (-1=none, 0=max abs int16, 1=taxicab, 2=euclidean, >2=p-norm) |
| -fa               | `"on"`, `"off"`, or `"auto"`                | `"auto"`                     | Enable/disable flash attention                        |
| --main-gpu        | integer, `"integrated"`, or `"dedicated"`   | —                            | GPU selection for multi-GPU systems                   |
| verbosity         | 0 – 3 (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG) | 0                            | Logging verbosity level                               |

#### IGPU/GPU  selection logic:

| Scenario                       | main-gpu not specified                | main-gpu: `"dedicated"`             | main-gpu: `"integrated"`           |
|---------------------------------|---------------------------------------|-------------------------------------|-------------------------------------|
| Devices considered              | All GPUs (dedicated + integrated)     | Only dedicated GPUs                 | Only integrated GPUs                |
| System with iGPU only           | ✅ Uses iGPU                          | ❌ Falls back to CPU                | ✅ Uses iGPU                        |
| System with dedicated GPU only  | ✅ Uses dedicated GPU                 | ✅ Uses dedicated GPU               | ❌ Falls back to CPU                |
| System with both                | ✅ Uses dedicated GPU (preferred)     | ✅ Uses dedicated GPU               | ✅ Uses integrated GPU              |


### 4. Instantiate the model

```js
const model = new GGMLBert(args)
```

### 5. Load the model

```js
await model.load()
```

`load()` takes no arguments. The addon streams each file listed in `files.model` directly from disk via `bare-fs` and then activates the model. There is no data loader, no progress callback, and no download step — the caller is responsible for ensuring the files already exist at the paths passed to the constructor.

### 6. Generate embeddings for input sequence

The model outputs a vector for the input sequence.

```js
const query = 'Hello, can you suggest a game I can play with my 1 year old daughter?'
const response = await model.run(query)
const embeddings = await response.await()
```

When `opts.stats` is enabled, `response.stats` includes runtime metrics such as `total_tokens`, `total_time_ms`, `tokens_per_second`, and `backendDevice` (`"cpu"` or `"gpu"`). `backendDevice` reflects the resolved device used at runtime after backend selection/fallback logic, not only the requested config.

### 7. Release Resources

Unload the model when finished:

```javascript
try {
  await model.unload()
} catch (error) {
  console.error('Failed to unload model:', error)
}
```

### API behavior by state

The following table describes the expected behavior of `run` and `cancel` depending on the current state (idle vs a job running). `cancel` can be called on the model (`model.cancel()`) or on the response (`response.cancel()`); both target the same underlying job.

| Current state | Action called | What happens |
|---------------|----------------|----------------------------------------------------------------|
| idle          | run            | **Allowed** — starts inference, returns `QvacResponse`        |
| idle          | cancel         | **Allowed** — no-op (no job to cancel); Promise resolves       |
| run           | run            | **Throw** — second `run()` throws "a job is already set or being processed" (can wait very briefly for previous job completion) |
| run           | cancel         | **Allowed** — cancels current job; Promise resolves when job has stopped      |

When `run()` is called while another job is active, the implementation first waits briefly for the previous job to settle. This preserves single-job behavior while still failing fast when the instance is busy. If the second run cannot be accepted (timeout or addon busy rejection), it throws:
- `"Cannot set new job: a job is already set or being processed"`

**Cancellation API:** Prefer cancelling from the model: `await model.cancel()`. This cancels the current job and the Promise resolves when the job has actually stopped (future-based in C++). You can also call `await response.cancel()` on the value returned by `run()`; it is equivalent and targets the same job. Both are no-op when idle.

## Quickstart Example

Clone the repository and navigate to it:
```bash
cd qvac-lib-infer-llamacpp-embed
```

Install dependencies:
```bash
npm install
```

Run the quickstart example (uses examples/quickstart.js):
```bash
npm run quickstart
```

## Other Examples

- [Batch Inference](./examples/batchInference.js) – Demonstrates running multiple prompts at once using batch inference.
- [Native Logging](./examples/nativelog.js) – Demonstrates C++ addon logging integration.

## Benchmarking

We conduct rigorous benchmarking of our embedding models to evaluate their retrieval effectiveness and computational efficiency across diverse tasks and datasets. Our evaluation framework incorporates standard information retrieval metrics and system performance indicators to provide a holistic view of model quality.

### Running Benchmarks

For instructions on running benchmarks yourself, see the [Benchmark Runner Documentation](./benchmarks/README.md).

The benchmarking covers:

* **Retrieval Quality**:

  * **nDCG\@k**: Quality of ranked results based on relevance and position
  * **MRR\@k**: Position of the first relevant result per query
  * **Recall\@k**: Coverage of relevant results in the top *k*
  * **Precision\@k**: Proportion of top *k* results that are relevant

Results are continuously updated with new releases to ensure up-to-date performance insights.

## Tests

Integration tests are located in [`test/integration/`](./test/integration/) and cover core functionality including model loading, inference, tool calling, multimodal capabilities, and configuration parameters.  
These tests help prevent regressions and ensure the library remains stable as contributions are made to the project.

Unit tests are located in [`test/unit/`](./test/unit/) and test the C++ addon components at a lower level, including backend selection, cache management, chat templates, context handling, and UTF8 token processing.  
These tests validate the native implementation and help catch issues early in development.

## Glossary

* **Bare Runtime** - Small and modular JavaScript runtime for desktop and mobile. [Learn more](https://docs.pears.com/reference/bare-overview).

## License

This project is licensed under the Apache-2.0 [License](./LICENSE) – see the LICENSE file for details.

_For questions or issues, please open an issue on the GitHub repository._
