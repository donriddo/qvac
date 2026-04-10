'use strict'

const QvacLogger = require('@qvac/logging')
const { createJobHandler, exclusiveRunQueue } = require('@qvac/infer-base')
const { SdInterface } = require('./addon')

const LOG_METHODS = ['error', 'warn', 'info', 'debug']

const RUN_BUSY_ERROR_MESSAGE = 'Cannot set new job: a job is already set or being processed'

/**
 * Text-to-image and image-to-image generation using stable-diffusion.cpp.
 * Supports SD1.x, SD2.x, SDXL, SD3, and FLUX.2 [klein].
 */
class ImgStableDiffusion {
  /**
   * @param {object} args
   * @param {object} args.files - Absolute file paths for model components
   * @param {string} args.files.model - Main model weights
   * @param {string} [args.files.clipL] - CLIP-L text encoder (FLUX.1 / SD3)
   * @param {string} [args.files.clipG] - CLIP-G text encoder (SDXL / SD3)
   * @param {string} [args.files.t5Xxl] - T5-XXL text encoder (FLUX.1 / SD3)
   * @param {string} [args.files.llm] - LLM text encoder (FLUX.2 klein)
   * @param {string} [args.files.vae] - VAE file
   * @param {object} args.config - SD context configuration (threads, device, type, etc.)
   * @param {object} [args.logger] - Structured logger
   * @param {object} [args.opts] - Optional inference options
   */
  constructor ({ files, config, logger = null, opts = {} }) {
    this._files = files
    this._config = config
    this.logger = new QvacLogger(logger)
    this.opts = opts
    this._job = createJobHandler({ cancel: () => this.addon.cancel() })
    this._run = exclusiveRunQueue()
    this.addon = null
    this._hasActiveResponse = false
    this._binding = null
    this._nativeLoggerActive = false
    this.state = { configLoaded: false }
  }

  async load () {
    if (this.state.configLoaded) {
      this.logger.info('Reload requested - unloading existing model first')
      await this.unload()
    }
    await this._load()
    this.state.configLoaded = true
  }

  async _load () {
    this.logger.info('Starting stable-diffusion model load')

    // Route the primary model file to the correct stable-diffusion.cpp param:
    //   path              — all-in-one checkpoints (SD1.x, SD2.x, SDXL, SD3 all-in-one GGUF)
    //   diffusionModelPath — standalone diffusion weights requiring separate encoders
    //                        (FLUX.2 klein → llm, SD3 pure GGUF → t5Xxl + clipL + clipG)
    const isSplitLayout = !!this._files.llm || !!this._files.t5Xxl
    const configurationParams = {
      path: isSplitLayout ? '' : (this._files.model || ''),
      diffusionModelPath: isSplitLayout ? (this._files.model || '') : '',
      clipLPath: this._files.clipL || '',
      clipGPath: this._files.clipG || '',
      t5XxlPath: this._files.t5Xxl || '',
      llmPath: this._files.llm || '',
      vaePath: this._files.vae || '',
      config: this._config
    }

    this.logger.info('Creating stable-diffusion addon with configuration:', configurationParams)
    this.addon = this._createAddon(configurationParams)

    this.logger.info('Activating stable-diffusion addon')
    await this.addon.activate()

    this.logger.info('Stable-diffusion model load completed successfully')
  }

  _createAddon (configurationParams) {
    this._binding = require('./binding')
    this._connectNativeLogger()
    return new SdInterface(
      this._binding,
      configurationParams,
      this._addonOutputCallback.bind(this)
    )
  }

  _connectNativeLogger () {
    if (!this._binding || !this.logger) return
    try {
      this._binding.setLogger((priority, message) => {
        const method = LOG_METHODS[priority] || 'info'
        if (typeof this.logger[method] === 'function') {
          this.logger[method](`[C++] ${message}`)
        }
      })
      this._nativeLoggerActive = true
    } catch (err) {
      this.logger.warn('Failed to connect native logger:', err.message)
    }
  }

  _releaseNativeLogger () {
    if (!this._nativeLoggerActive || !this._binding) return
    try {
      this._binding.releaseLogger()
    } catch (_) {}
    this._nativeLoggerActive = false
  }

  _addonOutputCallback (addon, event, data, error) {
    if (event.includes('Error')) {
      this.logger.error(`Job failed with error: ${error}`)
      this._job.fail(error)
      return
    }

    if (data instanceof Uint8Array || typeof data === 'string') {
      this._job.output(data)
      return
    }

    if (typeof data === 'object' && data !== null) {
      this._job.end(this.opts.stats ? data : null)
      return
    }

    this._job.output(data)
  }

  async run (params) {
    return this._run(() => this._runInternal(params))
  }

  async _runInternal (params) {
    if (!this.addon) {
      throw new Error('Addon not initialized. Call load() first.')
    }
    if (params.init_image) {
      throw new Error('img2img is not yet supported — omit init_image to run txt2img')
    }

    const mode = 'txt2img'
    this.logger.info('Starting generation with mode:', mode)

    if (this._hasActiveResponse) {
      throw new Error(RUN_BUSY_ERROR_MESSAGE)
    }

    const response = this._job.start()

    let accepted
    try {
      accepted = await this.addon.runJob({ ...params, mode })
    } catch (error) {
      this._job.fail(error)
      throw error
    }

    if (!accepted) {
      this._job.fail(new Error(RUN_BUSY_ERROR_MESSAGE))
      throw new Error(RUN_BUSY_ERROR_MESSAGE)
    }

    this._hasActiveResponse = true
    const finalized = response.await().finally(() => { this._hasActiveResponse = false })
    finalized.catch(() => {})
    response.await = () => finalized

    this.logger.info('Generation job started successfully')
    return response
  }

  async cancel () {
    if (this.addon) {
      await this.addon.cancel()
    }
  }

  async unload () {
    return this._run(async () => {
      await this.cancel()
      if (this._job.active) {
        this._job.fail(new Error('Model was unloaded'))
      }
      this._hasActiveResponse = false
      if (this.addon) {
        await this.addon.unload()
      }
      this._releaseNativeLogger()
      this.state.configLoaded = false
    })
  }

  getState () { return this.state }
}

module.exports = ImgStableDiffusion
