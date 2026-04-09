import type { QvacResponse } from '@qvac/infer-base'
import type QvacLogger from '@qvac/logging'

export type NumericLike = number | `${number}`

export interface Addon {
  activate(): Promise<void>
  runJob(params: GenerationParams & { mode: 'txt2img' | 'img2img' }): Promise<boolean>
  cancel(): Promise<void>
  unload(): Promise<void>
}

export type SamplerMethod =
  | 'euler'
  | 'euler_a'
  | 'heun'
  | 'dpm2'
  | 'dpm++2m'
  | 'dpm++2mv2'
  | 'dpm++2s_a'
  | 'lcm'
  | 'ipndm'
  | 'ipndm_v'
  | 'ddim_trailing'
  | 'tcd'
  | 'res_multistep'
  | 'res_2s'

export type WeightType =
  | 'auto'
  | 'f32'
  | 'f16'
  | 'bf16'
  | 'q2_k'
  | 'q3_k'
  | 'q4_0'
  | 'q4_1'
  | 'q4_k'
  | 'q5_0'
  | 'q5_1'
  | 'q5_k'
  | 'q6_k'
  | 'q8_0'

export type RngType = 'cpu' | 'cuda' | 'std_default'

export type ScheduleType =
  | 'discrete'
  | 'karras'
  | 'exponential'
  | 'ays'
  | 'gits'
  | 'sgm_uniform'
  | 'simple'
  | 'lcm'
  | 'smoothstep'
  | 'kl_optimal'
  | 'bong_tangent'

export type PredictionType = 'auto' | 'eps' | 'v' | 'edm_v' | 'flow' | 'flux_flow' | 'flux2_flow'

export type LoraApplyMode = 'auto' | 'immediately' | 'at_runtime'

export type CacheMode = 'disabled' | 'easycache' | 'ucache' | 'dbcache' | 'taylorseer' | 'cache-dit'

export interface SdConfig {
  threads?: NumericLike
  device?: 'gpu' | 'cpu'
  type?: WeightType
  rng?: RngType
  sampler_rng?: RngType
  clip_on_cpu?: boolean
  vae_on_cpu?: boolean
  vae_tiling?: boolean
  flash_attn?: boolean
  diffusion_fa?: boolean
  mmap?: boolean
  offload_to_cpu?: boolean
  prediction?: PredictionType
  flow_shift?: number
  diffusion_conv_direct?: boolean
  vae_conv_direct?: boolean
  force_sdxl_vae_conv_scale?: boolean
  backendsDir?: string
  tensor_type_rules?: string
  lora_apply_mode?: LoraApplyMode
  verbosity?: NumericLike
  [key: string]: string | number | boolean | undefined
}

export interface DiffusionFiles {
  model: string
  clipL?: string
  clipG?: string
  t5Xxl?: string
  llm?: string
  vae?: string
}

export interface ImgStableDiffusionArgs {
  files: DiffusionFiles
  config: SdConfig
  logger?: QvacLogger | Console | null
  opts?: { stats?: boolean }
}

export interface GenerationParams {
  prompt: string
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  guidance?: number
  sampling_method?: SamplerMethod
  sampler?: SamplerMethod
  scheduler?: ScheduleType
  seed?: number
  batch_count?: number
  vae_tiling?: boolean
  vae_tile_size?: number | string
  vae_tile_overlap?: number
  cache_mode?: CacheMode
  cache_preset?: string
  cache_threshold?: number
  eta?: number
  img_cfg_scale?: number
  clip_skip?: number
  init_image?: Uint8Array
  strength?: number
}

export interface RuntimeStats {
  modelLoadMs: number
  generationMs: number
  totalGenerationMs: number
  totalWallMs: number
  totalSteps: number
  totalGenerations: number
  totalImages: number
  totalPixels: number
  width: number
  height: number
  seed: number
}

export default class ImgStableDiffusion {
  protected addon: Addon | null
  opts: { stats?: boolean }
  logger: QvacLogger
  state: { configLoaded: boolean }

  constructor(args: ImgStableDiffusionArgs)

  load(): Promise<void>
  run(params: GenerationParams): Promise<QvacResponse>
  unload(): Promise<void>
  cancel(): Promise<void>
  getState(): { configLoaded: boolean }
}

export { QvacResponse, RuntimeStats }
