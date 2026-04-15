'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const QvacLogger = require('@qvac/logging')
const { createJobHandler, exclusiveRunQueue } = require('@qvac/infer-base')
const { LlamaInterface, mapAddonEvent } = require('./addon')

const RUN_BUSY_ERROR_MESSAGE = 'Cannot set new job: a job is already set or being processed'

function normalizeRunOptions (runOptions) {
  if (runOptions === undefined) {
    return { prefill: false, generationParams: undefined, cacheKey: undefined, saveCacheToDisk: false }
  }

  if (!runOptions || typeof runOptions !== 'object' || Array.isArray(runOptions)) {
    throw new TypeError('Run options must be an object when provided')
  }

  if (runOptions.prefill !== undefined &&
      typeof runOptions.prefill !== 'boolean') {
    throw new TypeError('prefill must be a boolean when provided')
  }

  if (runOptions.generationParams !== undefined &&
      (typeof runOptions.generationParams !== 'object' || runOptions.generationParams === null || Array.isArray(runOptions.generationParams))) {
    throw new TypeError('generationParams must be a plain object when provided')
  }

  if (runOptions.cacheKey !== undefined && typeof runOptions.cacheKey !== 'string') {
    throw new TypeError('cacheKey must be a string when provided')
  }

  if (runOptions.saveCacheToDisk !== undefined && typeof runOptions.saveCacheToDisk !== 'boolean') {
    throw new TypeError('saveCacheToDisk must be a boolean when provided')
  }

  return {
    prefill: runOptions.prefill === true,
    generationParams: runOptions.generationParams,
    cacheKey: runOptions.cacheKey,
    saveCacheToDisk: runOptions.saveCacheToDisk === true
  }
}

const VALIDATION_TYPES = ['none', 'split', 'dataset']
const DEFAULT_VALIDATION_FRACTION = 0.05

function normalizeFinetuneParams (opts) {
  const validation = opts.validation
  if (Object.prototype.hasOwnProperty.call(opts, 'evalDatasetPath')) {
    throw new Error(
      "Top-level evalDatasetPath is no longer supported. Use validation.path with validation.type set to 'dataset'."
    )
  }
  if (validation == null || typeof validation !== 'object' || !('type' in validation)) {
    throw new Error(
      'Finetuning options must include validation: { type: \'none\' | \'split\' | \'dataset\'[, fraction?: number][, path?: string] }. ' +
      'Example: validation: { type: \'split\', fraction: 0.05 }, validation: { type: \'dataset\', path: \'./eval.jsonl\' }, or validation: { type: \'none\' }.'
    )
  }
  const out = { ...opts }
  const type = validation.type
  if (!VALIDATION_TYPES.includes(type)) {
    throw new Error(
      `validation.type must be one of ${VALIDATION_TYPES.join(', ')}; got: ${type}`
    )
  }
  if (type === 'none') {
    out.validationSplit = 0
    out.useEvalDatasetForValidation = false
    delete out.evalDatasetPath
  } else if (type === 'split') {
    const fraction = validation.fraction ?? DEFAULT_VALIDATION_FRACTION
    out.validationSplit = Math.max(0, Math.min(1, Number(fraction)))
    out.useEvalDatasetForValidation = false
    delete out.evalDatasetPath
  } else {
    const evalPath = validation.path
    if (!evalPath || typeof evalPath !== 'string' || evalPath.trim() === '') {
      throw new Error(
        "validation.type is 'dataset' but no path is provided. Set validation.path to the eval dataset file path (e.g. validation: { type: 'dataset', path: './eval.jsonl' })."
      )
    }
    if (evalPath === opts.trainDatasetDir) {
      throw new Error(
        "validation.type is 'dataset' but validation.path is the same as trainDatasetDir. Provide a separate eval dataset path."
      )
    }
    out.evalDatasetPath = evalPath
    out.validationSplit = 0
    out.useEvalDatasetForValidation = true
  }
  delete out.validation
  return out
}

/**
 * Picks the primary GGUF path from an ordered file list.
 *
 * For sharded models the caller passes
 * `[tensors.txt, shard-00001-of-N.gguf, ..., shard-N-of-N.gguf]`.
 * The first entry matching the shard regex is returned so the value matches
 * the C++ `GGUFShards::expandGGUFIntoShards` regex contract.
 * For non-sharded single-file models the only entry is returned.
 *
 * @param {string[]} files - ordered array of absolute paths
 * @returns {string} the primary GGUF path
 */
function pickPrimaryGgufPath (files) {
  const SHARD_REGEX = /-\d+-of-\d+\.gguf$/
  return files.find((p) => SHARD_REGEX.test(p)) || files[0]
}

class LlmLlamacpp {
  constructor ({ files, config, logger = null, opts = {} }) {
    if (!files || !Array.isArray(files.model) || files.model.length === 0) {
      throw new TypeError('files.model must be a non-empty array of absolute paths')
    }
    this._files = files.model
    this._projectionModelPath = files.projectionModel || ''
    this._config = config
    this.logger = new QvacLogger(logger)
    this.opts = opts
    // The cancel closure dereferences `this.addon` lazily, so it is safe even though
    // `this.addon` is `null` at construction time — it is only invoked from
    // `response.cancel()` after `_load()` has assigned the addon. The optional chain
    // also makes a stale `response.cancel()` after `unload()` a no-op.
    this._job = createJobHandler({ cancel: () => this.addon?.cancel() })
    this._run = exclusiveRunQueue()
    this.addon = null
    this._checkpointSaveDir = null
    this._hasActiveResponse = false
    // Stateful flag carried across `mapAddonEvent` calls so the post-finetune
    // TPS trailer the C++ addon emits is not mistaken for a fresh inference
    // result. Lives on the model so unit tests can poke at it.
    this._addonEventState = { skipNextRuntimeStats: false }
    this.state = { configLoaded: false }
  }

  async load () {
    if (this.state.configLoaded) return
    await this._load()
    this.state.configLoaded = true
  }

  async _load () {
    this.logger.info('Starting model load')
    const primaryGgufPath = pickPrimaryGgufPath(this._files)
    const configurationParams = {
      path: primaryGgufPath,
      projectionPath: this._projectionModelPath,
      config: { ...this._config }
    }
    this.addon = this._createAddon(configurationParams)

    try {
      if (this._files.length > 1) {
        await this._streamShards()
      }
      await this.addon.activate()
    } catch (loadError) {
      // Best-effort cleanup of the partially-initialized addon so a subsequent
      // load() does not leak a zombie native instance.
      try { await this.addon?.unload?.() } catch (_) {}
      this.addon = null
      throw loadError
    }
    this.logger.info('Model load completed successfully')
  }

  async _streamShards () {
    for (const filePath of this._files) {
      const filename = path.basename(filePath)
      const stream = fs.createReadStream(filePath)
      for await (const chunk of stream) {
        await this.addon.loadWeights({ filename, chunk, completed: false })
      }
      await this.addon.loadWeights({ filename, chunk: null, completed: true })
      this.logger.info(`Streamed weights for ${filename}`)
    }
  }

  /**
   * Public API entrypoint for inference.
   * @param {Message[]} prompt - Input prompt array of messages
   * @param {RunOptions} [runOptions] - Optional run settings (prefill, generationParams, cacheKey, saveCacheToDisk)
   * @returns {Promise<QvacResponse>}
   */
  async run (prompt, runOptions = {}) {
    return this._run(() => this._runInternal(prompt, runOptions))
  }

  async _runInternal (prompt, runOptions = {}) {
    if (!this.addon) {
      throw new Error('Addon not initialized. Call load() first.')
    }
    if (this._hasActiveResponse) {
      throw new Error(RUN_BUSY_ERROR_MESSAGE)
    }

    if (!Array.isArray(prompt)) {
      throw new TypeError('Prompt input must be Message[]')
    }
    const { prefill, generationParams, cacheKey, saveCacheToDisk } = normalizeRunOptions(runOptions)

    this.logger.info('Starting inference with prompt:', prompt)

    const textMessages = []
    const mediaItems = []

    for (const message of prompt) {
      if (message.role === 'user' &&
          message.type === 'media' &&
          message.content instanceof Uint8Array) {
        mediaItems.push(message.content)
        textMessages.push({ ...message, content: '' })
      } else {
        textMessages.push(message)
      }
    }

    const promptMessages = []

    for (const mediaData of mediaItems) {
      promptMessages.push({ type: 'media', content: mediaData })
    }

    promptMessages.push({
      type: 'text',
      input: JSON.stringify(textMessages),
      prefill,
      generationParams,
      cacheKey,
      saveCacheToDisk
    })

    const response = this._job.start()

    let accepted
    try {
      accepted = await this.addon.runJob(promptMessages)
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
    finalized.catch((err) => {
      this.logger?.warn?.('Inference response rejected:', err?.message || err)
    })
    response.await = () => finalized

    this.logger.info('Inference job started successfully')
    return response
  }

  async finetune (finetuningOptions = undefined) {
    if (!this.addon) {
      throw new Error('Addon not initialized. Call load() first.')
    }
    if (!finetuningOptions) {
      throw new Error('Finetuning parameters are required.')
    }
    if (finetuningOptions.checkpointSaveDir) {
      this._checkpointSaveDir = finetuningOptions.checkpointSaveDir
    }
    const paramsToSend = normalizeFinetuneParams(finetuningOptions)
    this.logger.info('finetune() called')
    this.logger.info('Finetuning parameters:', finetuningOptions)

    return this._run(async () => {
      if (this._hasActiveResponse) {
        throw new Error(RUN_BUSY_ERROR_MESSAGE)
      }

      const response = this._job.start()
      let accepted
      try {
        accepted = await this.addon.finetune(paramsToSend)
      } catch (err) {
        this._job.fail(err)
        throw err
      }

      if (!accepted) {
        this._job.fail(new Error(RUN_BUSY_ERROR_MESSAGE))
        throw new Error(RUN_BUSY_ERROR_MESSAGE)
      }

      this._hasActiveResponse = true
      const finalized = response.await().finally(() => { this._hasActiveResponse = false })
      finalized.catch((err) => {
        this.logger?.warn?.('Finetune response rejected:', err?.message || err)
      })
      response.await = () => finalized
      return response
    })
  }

  _handleAddonOutputEvent (eventType, data, error) {
    if (eventType === 'LogMsg') {
      const logMsg = typeof data === 'string' ? data : (data?.message || JSON.stringify(data))
      this.logger?.info?.(logMsg)
      return
    }

    if (eventType === 'Error') {
      this.logger.error('Job failed with error:', error)
      this._job.fail(error)
    } else if (eventType === 'Output') {
      this._job.output(data)
    } else if (eventType === 'FinetuneProgress') {
      if (this.opts.stats && data && data.stats) {
        this._job.active?.updateStats(data.stats)
      }
    } else if (eventType === 'JobEnded') {
      this.logger.info('Job completed')
      const isFinetuneTerminal = data && typeof data === 'object' && data.op === 'finetune' && typeof data.status === 'string'
      if (isFinetuneTerminal) {
        this._job.end(null, data)
      } else {
        this._job.end(this.opts.stats ? data : null)
      }
    }
  }

  _addonOutputCallback (addon, event, data, error) {
    // Event-name normalization lives in `addon.js` (`mapAddonEvent`) so the
    // native binding wrapper owns the C++ event vocabulary. This shim only
    // forwards the resulting logical event into `_handleAddonOutputEvent`.
    const mapped = mapAddonEvent(event, data, error, this._addonEventState)
    if (mapped === null) return
    this._handleAddonOutputEvent(mapped.type, mapped.data, mapped.error)
  }

  _createAddon (configurationParams) {
    const binding = require('./binding')
    return new LlamaInterface(
      binding,
      configurationParams,
      this._addonOutputCallback.bind(this)
    )
  }

  async pause () {
    if (this.addon?.cancel) {
      await this.addon.cancel()
    }
  }

  async cancel () {
    if (this.addon?.cancel) {
      await this.addon.cancel()
    }
    this._clearPauseCheckpoints()
  }

  _clearPauseCheckpoints () {
    const checkpointDir = this._checkpointSaveDir
    if (!checkpointDir) return
    try {
      const entries = fs.readdirSync(checkpointDir, { withFileTypes: true })
      for (const entry of entries) {
        if (entry.isDirectory() && entry.name.startsWith('pause_checkpoint_step_')) {
          fs.rmSync(path.join(checkpointDir, entry.name), { recursive: true, force: true })
        }
      }
    } catch (err) {
      this.logger.error('Failed to clear pause checkpoints:', err)
    }
  }

  async unload () {
    return this._run(async () => {
      try {
        await this.pause()
      } catch (_) {}
      if (this._job.active) {
        this._job.fail(new Error('Model was unloaded'))
      }
      this._hasActiveResponse = false
      if (this.addon) {
        await this.addon.unload()
        // Null the addon reference so post-unload `cancel()` / `run()` calls hit the
        // `if (!this.addon)` guard instead of dereferencing a disposed native handle.
        this.addon = null
      }
      this.state.configLoaded = false
    })
  }

  getState () { return this.state }
}

module.exports = LlmLlamacpp
module.exports.pickPrimaryGgufPath = pickPrimaryGgufPath
