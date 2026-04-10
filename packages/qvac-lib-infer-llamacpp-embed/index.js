'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const QvacLogger = require('@qvac/logging')
const { createJobHandler, exclusiveRunQueue } = require('@qvac/infer-base')
const { BertInterface } = require('./addon')

const RUN_BUSY_ERROR_MESSAGE = 'Cannot set new job: a job is already set or being processed'

const SHARD_REGEX = /-\d+-of-\d+\.gguf$/

class GGMLBert {
  constructor ({ files, config = {}, logger = null, opts = {} }) {
    if (!files || !Array.isArray(files.model) || files.model.length === 0) {
      throw new TypeError('files.model must be a non-empty array of absolute paths')
    }
    this._files = files.model
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
    this._hasActiveResponse = false
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
    this.logger.info('Starting model load')
    // Pick the primary GGUF path that goes into `configurationParams.path`. For
    // sharded models the caller passes [tensors.txt, shard-00001-of-NNNNN.gguf, ..., shard-NNNNN-of-NNNNN.gguf];
    // we pass the FIRST entry that matches the shard regex (skipping `tensors.txt` at index 0)
    // so the value matches the C++ `GGUFShards::expandGGUFIntoShards` regex contract.
    // For non-sharded single-file models there is only one entry and the find returns it.
    const primaryGgufPath = this._files.find((p) => SHARD_REGEX.test(p)) || this._files[0]
    const configurationParams = {
      path: primaryGgufPath,
      config: this._config
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

  async run (input) {
    return this._run(() => this._runInternal(input))
  }

  async _runInternal (text) {
    if (!this.addon) {
      throw new Error('Addon not initialized. Call load() first.')
    }
    if (this._hasActiveResponse) {
      throw new Error(RUN_BUSY_ERROR_MESSAGE)
    }

    this.logger.info('Starting inference embeddings for text:', text)
    const inputData = Array.isArray(text)
      ? { type: 'sequences', input: text }
      : { type: 'text', input: text }

    const response = this._job.start()

    let accepted
    try {
      accepted = await this.addon.runJob(inputData)
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
    return response
  }

  _addonOutputCallback (addon, event, data, error) {
    const isStatsData = typeof data === 'object' && data !== null && (
      'tokens_per_second' in data ||
      ('total_tokens' in data || 'total_time_ms' in data || 'batch_size' in data || 'context_size' in data)
    )
    if (isStatsData) {
      const runtimeStats = { ...data }
      if (runtimeStats.backendDevice === 0) {
        runtimeStats.backendDevice = 'cpu'
      } else if (runtimeStats.backendDevice === 1) {
        runtimeStats.backendDevice = 'gpu'
      }
      this._job.end(this.opts.stats ? runtimeStats : null)
      return
    }

    if (event.includes('Error')) {
      this.logger.error('Job failed with error:', error)
      this._job.fail(error)
      return
    }

    if (event.includes('Embeddings')) {
      this._job.output(data)
      return
    }

    // Unknown event type — log it instead of feeding the payload into the active
    // response output stream as if it were embedding data. The native layer is
    // expected to emit only `Embeddings`, `Error`, or stats; reaching this branch
    // indicates a native-layer change worth surfacing.
    this.logger.debug(`Unhandled addon event: ${event} (data type: ${typeof data})`)
  }

  _createAddon (configurationParams) {
    const binding = require('./binding')
    return new BertInterface(
      binding,
      configurationParams,
      this._addonOutputCallback.bind(this)
    )
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
        // Null the addon reference so post-unload `cancel()` / `run()` calls hit the
        // `if (!this.addon)` guard instead of dereferencing a disposed native handle.
        this.addon = null
      }
      this.state.configLoaded = false
    })
  }

  async cancel () {
    if (this.addon?.cancel) {
      await this.addon.cancel()
    }
  }

  getState () { return this.state }
}

module.exports = GGMLBert
