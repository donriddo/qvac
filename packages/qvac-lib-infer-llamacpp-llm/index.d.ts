import type { QvacResponse } from '@qvac/infer-base'
import type QvacLogger from '@qvac/logging'

export type NumericLike = number | `${number}`

export interface AddonMessage {
  type: 'text'
  input: string
  prefill?: boolean
  /**
   * Per-call sampling overrides forwarded by `LlmLlamacpp.run()` from
   * `RunOptions.generationParams`. Carried on the `text` message and consumed
   * by the native binding so each `runJob` can use a different temp / top_p /
   * seed / etc. without re-loading the model.
   */
  generationParams?: GenerationParams
}
export interface AddonMediaMessage {
  type: 'media'
  content: Uint8Array
}
export type AddonRunJobMessage = AddonMessage | AddonMediaMessage

export interface Addon {
  loadWeights(data: { filename: string; chunk: Uint8Array | null; completed: boolean }, logger?: QvacLogger): Promise<void>
  activate(): Promise<void>
  runJob(data: AddonRunJobMessage[]): Promise<boolean>
  cancel(): Promise<void>
  finetune?(params: FinetuneOptions): Promise<boolean>
  unload(): Promise<void>
}

export interface LlamaConfig {
  device?: string
  gpu_layers?: NumericLike
  ctx_size?: NumericLike
  system_prompt?: string
  lora?: string
  temp?: NumericLike
  top_p?: NumericLike
  top_k?: NumericLike
  predict?: NumericLike
  seed?: NumericLike
  no_mmap?: boolean | ''
  reverse_prompt?: string
  repeat_penalty?: NumericLike
  presence_penalty?: NumericLike
  frequency_penalty?: NumericLike
  tools?: boolean | string
  verbosity?: NumericLike
  n_discarded?: NumericLike
  'main-gpu'?: NumericLike | string
  [key: string]: string | number | boolean | string[] | undefined
}

export interface LlmLlamacppArgs {
  files: { model: string[]; projectionModel?: string }
  config: LlamaConfig
  logger?: QvacLogger | Console | null
  opts?: { stats?: boolean }
}

export interface UserTextMessage {
  role: 'system' | 'assistant' | 'user' | 'tool' | 'session' | string
  content: string
  type?: undefined
  [key: string]: any
}

export interface UserMediaMessage {
  role: 'user'
  type: 'media'
  content: Uint8Array
}

export interface ChatFunctionDefinition {
  type: 'function'
  name: string
  description?: string
  parameters?: Record<string, any>
}

export type Message =
  | UserTextMessage
  | UserMediaMessage
  | ChatFunctionDefinition

export interface GenerationParams {
  temp?: number
  top_p?: number
  top_k?: number
  predict?: number
  seed?: number
  frequency_penalty?: number
  presence_penalty?: number
  repeat_penalty?: number
}

export interface RunOptions {
  prefill?: boolean
  generationParams?: GenerationParams
}

export interface RuntimeStats {
  TTFT: number
  TPS: number
  CacheTokens: number
  generatedTokens: number
  promptTokens: number
  contextSlides: number
  backendDevice: 'cpu' | 'gpu'
}

export interface FinetuneValidationNone {
  type: 'none'
}

export interface FinetuneValidationSplit {
  type: 'split'
  fraction?: number
}

export interface FinetuneValidationDataset {
  type: 'dataset'
  path: string
}

export type FinetuneValidation =
  | FinetuneValidationNone
  | FinetuneValidationSplit
  | FinetuneValidationDataset

export interface FinetuneOptions {
  /** Path to the training dataset file (e.g. `.jsonl` for SFT, `.txt` for causal). */
  trainDatasetDir: string
  /**
   * How to run validation. Required — there is no default.
   * `{ type: 'none' }` disables validation. `{ type: 'split', fraction? }` reserves
   * a fraction of the training data (default 0.05). `{ type: 'dataset', path }`
   * uses a separate eval dataset file.
   */
  validation: FinetuneValidation
  /** Directory (or file path) where the final LoRA adapter will be written. */
  outputParametersDir: string
  /** Number of training epochs. Default 1. */
  numberOfEpochs?: number
  /** Initial learning rate. Default 1e-4. */
  learningRate?: number
  /** Training sequence length (tokens). Default 128. */
  contextLength?: number
  /**
   * Backend `n_batch` (number of tokens processed per batch). Must be `>= microBatchSize`
   * and divisible by it when both are set. Default 128.
   */
  batchSize?: number
  /**
   * Backend `n_ubatch` (micro-batch size). Adjusted to gcd(datasetSampleCount, requested)
   * if needed. Must be `<= batchSize` when both are set. Default 128.
   */
  microBatchSize?: number
  /** Use SFT (chat) mode if `true`, causal mode otherwise. Default `false`. */
  assistantLossOnly?: boolean
  /**
   * Comma-separated target modules (e.g. `attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down,output`,
   * or `all`). Default attention Q, K, V, O only.
   */
  loraModules?: string
  /** LoRA rank. Default 8. */
  loraRank?: number
  /** LoRA alpha (scaling). Default 16.0. */
  loraAlpha?: number
  /** LoRA init standard deviation. Default 0.02. */
  loraInitStd?: number
  /** Seed for LoRA weight initialization (0 = non-deterministic). Default 42. */
  loraSeed?: number
  /** Directory where checkpoints (and pause checkpoints) are saved. Default `./checkpoints`. */
  checkpointSaveDir?: string
  /** Save a checkpoint every N steps (0 = pause checkpoints only). Default 0. */
  checkpointSaveSteps?: number
  /** Path to a chat template file (used in SFT mode). Default `""`. */
  chatTemplatePath?: string
  /** Learning-rate schedule. Default `"cosine"`. */
  lrScheduler?: 'constant' | 'cosine' | 'linear'
  /** Minimum learning rate (used by cosine/linear schedulers). Default 0. */
  lrMin?: number
  /** Warmup ratio (0–1). Requires `warmupRatioSet: true` to take effect. Default 0.1. */
  warmupRatio?: number
  /** When `true`, warmup steps = `warmupRatio × totalSteps`. Default `false`. */
  warmupRatioSet?: boolean
  /** Explicit warmup steps (used when `warmupStepsSet: true`). Default 0. */
  warmupSteps?: number
  /** When `true`, use `warmupSteps` directly instead of `warmupRatio`. Default `false`. */
  warmupStepsSet?: boolean
  /** Optimizer weight decay. Default 0.01. */
  weightDecay?: number
}

export interface FinetuneProgressStats {
  is_train: boolean
  loss: number
  loss_uncertainty: number
  accuracy: number
  accuracy_uncertainty: number
  global_steps: number
  current_epoch: number
  current_batch: number
  total_batches: number
  elapsed_ms: number
  eta_ms: number
}

export interface FinetuneHandle {
  on(event: 'stats', cb: (stats: FinetuneProgressStats) => void): this
  removeListener(event: 'stats', cb: (stats: FinetuneProgressStats) => void): this
  await(): Promise<FinetuneResult>
}

export interface FinetuneStats {
  train_loss?: number
  train_loss_uncertainty?: number
  val_loss?: number
  val_loss_uncertainty?: number
  train_accuracy?: number
  train_accuracy_uncertainty?: number
  val_accuracy?: number
  val_accuracy_uncertainty?: number
  learning_rate?: number
  global_steps: number
  epochs_completed: number
}

export interface FinetuneResult {
  op: 'finetune'
  status: 'COMPLETED' | 'PAUSED'
  stats?: FinetuneStats
}

export default class LlmLlamacpp {
  protected addon: Addon | null
  opts: { stats?: boolean }
  logger: QvacLogger
  state: { configLoaded: boolean }

  constructor(args: LlmLlamacppArgs)

  load(): Promise<void>
  run(prompt: Message[], runOptions?: RunOptions): Promise<QvacResponse>
  finetune(finetuningOptions: FinetuneOptions): Promise<FinetuneHandle>
  cancel(): Promise<void>
  pause(): Promise<void>
  unload(): Promise<void>
  getState(): { configLoaded: boolean }
}

export { QvacResponse, FinetuneHandle, FinetuneProgressStats, FinetuneOptions, FinetuneValidation }
