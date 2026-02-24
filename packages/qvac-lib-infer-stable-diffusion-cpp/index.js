'use strict'

const path = require('bare-path')

const BaseInference = require('@qvac/infer-base/WeightsProvider/BaseInference')
const WeightsProvider = require('@qvac/infer-base/WeightsProvider/WeightsProvider')
const { SdInterface } = require('./addon')

const noop = () => {}

/** Max ms to wait for the previous job to finish before throwing. */
const PREVIOUS_JOB_WAIT_MS = 30
const RUN_BUSY_ERROR_MESSAGE = 'Cannot set new job: a job is already set or being processed'

/**
 * Image and video generation using stable-diffusion.cpp.
 * Supports SD1.x, SD2.x, SDXL, SD3, FLUX, Wan2.x video models.
 */
class ImgStableDiffusion extends BaseInference {
  /**
   * @param {object} args
   * @param {object} args.loader - Data loader (Hyperdrive, filesystem, etc.)
   * @param {object} [args.logger] - Structured logger
   * @param {object} [args.opts] - Optional inference options
   * @param {string} [args.diskPath='.'] - Local directory for downloaded weights
   * @param {string} args.modelName - Model file name (e.g. 'flux1-dev-q4_0.gguf')
   * @param {string} [args.clipLModel] - Optional CLIP-L model file name (FLUX.1 / SD3)
   * @param {string} [args.clipGModel] - Optional CLIP-G model file name (SDXL / SD3)
   * @param {string} [args.t5XxlModel] - Optional T5-XXL text encoder file name (FLUX.1 / SD3)
   * @param {string} [args.llmModel] - Optional LLM text encoder file name (FLUX.2 klein → Qwen3 8B)
   * @param {string} [args.vaeModel] - Optional VAE file name
   * @param {object} config - SD context configuration (threads, device, wtype, etc.)
   */
  constructor (
    {
      opts = {},
      loader,
      logger = null,
      diskPath = '.',
      modelName,
      clipLModel,
      clipGModel,
      t5XxlModel,
      llmModel,
      vaeModel
    },
    config
  ) {
    super({ logger, opts })
    this._config = config
    this._diskPath = diskPath
    this._modelName = modelName
    this._clipLModel = clipLModel || null
    this._clipGModel = clipGModel || null
    this._t5XxlModel = t5XxlModel || null
    this._llmModel = llmModel || null
    this._vaeModel = vaeModel || null
    this.weightsProvider = new WeightsProvider(loader, this.logger)
    this._lastJobResult = Promise.resolve()
  }

  /**
   * Load model weights, initialize the native addon, and activate.
   * @param {boolean} [closeLoader=true]
   * @param {Function} [onDownloadProgress]
   */
  async _load (closeLoader = true, onDownloadProgress = noop) {
    this.logger.info('Starting stable-diffusion model load')

    try {
      const filesToDownload = [this._modelName]
      if (this._clipLModel) filesToDownload.push(this._clipLModel)
      if (this._clipGModel) filesToDownload.push(this._clipGModel)
      if (this._t5XxlModel) filesToDownload.push(this._t5XxlModel)
      if (this._llmModel) filesToDownload.push(this._llmModel)
      if (this._vaeModel) filesToDownload.push(this._vaeModel)

      await this.weightsProvider.downloadFiles(filesToDownload, this._diskPath, {
        closeLoader,
        onDownloadProgress
      })

      const configurationParams = {
        path: path.join(this._diskPath, this._modelName),
        clipLPath: this._clipLModel ? path.join(this._diskPath, this._clipLModel) : '',
        clipGPath: this._clipGModel ? path.join(this._diskPath, this._clipGModel) : '',
        t5XxlPath: this._t5XxlModel ? path.join(this._diskPath, this._t5XxlModel) : '',
        llmPath: this._llmModel ? path.join(this._diskPath, this._llmModel) : '',
        vaePath: this._vaeModel ? path.join(this._diskPath, this._vaeModel) : '',
        config: this._config
      }

      this.logger.info('Creating stable-diffusion addon with configuration:', configurationParams)
      this.addon = this._createAddon(configurationParams)

      this.logger.info('Activating stable-diffusion addon')
      await this.addon.activate()

      this.logger.info('Stable-diffusion model load completed successfully')
    } catch (error) {
      this.logger.error('Error during stable-diffusion model load:', error)
      throw error
    }
  }

  /**
   * @param {Function} [onDownloadProgress]
   * @param {object} [opts]
   */
  async _downloadWeights (onDownloadProgress, opts) {
    const filesToDownload = [this._modelName]
    if (this._clipLModel) filesToDownload.push(this._clipLModel)
    if (this._clipGModel) filesToDownload.push(this._clipGModel)
    if (this._t5XxlModel) filesToDownload.push(this._t5XxlModel)
    if (this._llmModel) filesToDownload.push(this._llmModel)
    if (this._vaeModel) filesToDownload.push(this._vaeModel)

    return this.weightsProvider.downloadFiles(filesToDownload, this._diskPath, {
      closeLoader: opts.closeLoader,
      onDownloadProgress
    })
  }

  /**
   * @param {object} configurationParams
   * @returns {SdInterface}
   */
  _createAddon (configurationParams) {
    const binding = require('./binding')
    return new SdInterface(
      binding,
      configurationParams,
      this._addonOutputCallback.bind(this)
    )
  }

  _addonOutputCallback (addon, event, data, error) {
    if (typeof data === 'object' && data !== null && 'generation_time' in data) {
      return this._outputCallback(addon, 'JobEnded', 'OnlyOneJob', data, null)
    }

    let mappedEvent = event
    if (event.includes('Error')) {
      mappedEvent = 'Error'
    } else if (data instanceof Uint8Array) {
      mappedEvent = 'Output'
    } else if (typeof data === 'string') {
      try {
        const parsed = JSON.parse(data)
        if ('step' in parsed && 'total' in parsed) {
          mappedEvent = 'StepProgress'
        }
      } catch (_) {
        mappedEvent = 'Output'
      }
    }

    return this._outputCallback(addon, mappedEvent, 'OnlyOneJob', data, error)
  }

  /**
   * Cancel the current generation job.
   */
  async cancel () {
    if (this.addon?.cancel) {
      await this.addon.cancel()
    }
  }

  /**
   * Unload the model and release all resources.
   */
  async unload () {
    return this._withExclusiveRun(async () => {
      await this.cancel()
      const currentJobResponse = this._jobToResponse.get('OnlyOneJob')
      if (currentJobResponse) {
        currentJobResponse.failed(new Error('Model was unloaded'))
        this._deleteJobMapping('OnlyOneJob')
      }
      await super.unload()
    })
  }

  /**
   * Generate an image from text.
   * @param {object} params - Generation parameters
   * @param {string} params.prompt
   * @param {string} [params.negative_prompt]
   * @param {number} [params.width=512]
   * @param {number} [params.height=512]
   * @param {number} [params.steps=20]
   * @param {number} [params.cfg_scale=7.0]
   * @param {string} [params.sampler='euler_a']
   * @param {number} [params.seed=-1]
   * @param {number} [params.batch_count=1]
   * @returns {Promise<QvacResponse>}
   */
  async txt2img (params) {
    return this._runGeneration({ ...params, mode: 'txt2img' })
  }

  /**
   * Generate an image from an input image and text.
   * @param {object} params
   * @param {Uint8Array} params.init_image - Input image bytes (PNG/JPEG)
   * @param {number} [params.strength=0.75] - Denoising strength (0.0-1.0)
   * @returns {Promise<QvacResponse>}
   */
  async img2img (params) {
    if (!params.init_image) {
      throw new Error('img2img requires init_image parameter')
    }
    return this._runGeneration({ ...params, mode: 'img2img' })
  }

  /**
   * Generate a video from text (requires Wan2.x or similar video model).
   * @param {object} params
   * @param {string} params.prompt
   * @param {number} [params.frames=16]
   * @param {number} [params.fps=8]
   * @returns {Promise<QvacResponse>}
   */
  async txt2vid (params) {
    return this._runGeneration({ ...params, mode: 'txt2vid' })
  }

  async _runGeneration (params) {
    this.logger.info('Starting generation with mode:', params.mode)

    return this._withExclusiveRun(async () => {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          reject(new Error(RUN_BUSY_ERROR_MESSAGE))
        }, PREVIOUS_JOB_WAIT_MS)
        this._lastJobResult
          .then(() => { clearTimeout(timer); resolve() })
          .catch(() => { clearTimeout(timer); resolve() })
      })

      const response = this._createResponse('OnlyOneJob')

      let accepted
      try {
        accepted = await this.addon.runJob(params)
      } catch (error) {
        this._deleteJobMapping('OnlyOneJob')
        response.failed(error)
        throw error
      }

      if (!accepted) {
        this._deleteJobMapping('OnlyOneJob')
        const msg = RUN_BUSY_ERROR_MESSAGE
        response.failed(new Error(msg))
        throw new Error(msg)
      }

      this._lastJobResult = response.await()

      this.logger.info('Generation job started successfully')

      return response
    })
  }
}

module.exports = ImgStableDiffusion
