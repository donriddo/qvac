const path = require('bare-path')

/**
 * Map a raw native event from the C++ embed addon to a logical event.
 *
 * The native binding emits events with C++-mangled names and varied
 * payload shapes. This wrapper normalizes them into one of:
 *   - `'Output'`     — embeddings payload (`Embeddings` family event)
 *   - `'Error'`      — failure
 *   - `'JobEnded'`   — terminal RuntimeStats payload (with `backendDevice`
 *                       mapped from `0/1` to `'cpu'/'gpu'`)
 *
 * Returns `{ type, data, error }` or `null` if the event should be
 * dropped (currently never — embed has no skip-flag state, but the
 * shape mirrors the LLM addon for consistency).
 *
 *
 * @param {string} rawEvent
 * @param {*} rawData
 * @param {*} rawError
 * @returns {{ type: string, data: *, error: * } | null}
 */
function mapAddonEvent (rawEvent, rawData, rawError) {
  // RuntimeStats: structurally detected so we don't couple to C++ key
  // ordering. The embed addon emits these as the terminal event for a
  // job (`tokens_per_second` is the marker; `total_tokens` /
  // `total_time_ms` / `batch_size` / `context_size` are the other
  // canonical fields).
  const isStatsData =
    rawData &&
    typeof rawData === 'object' &&
    (
      'tokens_per_second' in rawData ||
      'total_tokens' in rawData ||
      'total_time_ms' in rawData ||
      'batch_size' in rawData ||
      'context_size' in rawData
    )
  if (isStatsData) {
    const stats = { ...rawData }
    if (stats.backendDevice === 0) {
      stats.backendDevice = 'cpu'
    } else if (stats.backendDevice === 1) {
      stats.backendDevice = 'gpu'
    }
    return { type: 'JobEnded', data: stats, error: null }
  }

  if (typeof rawEvent === 'string' && rawEvent.includes('Error')) {
    return { type: 'Error', data: rawData, error: rawError }
  }

  if (typeof rawEvent === 'string' && rawEvent.includes('Embeddings')) {
    return { type: 'Output', data: rawData, error: null }
  }

  return null
}

/// An interface between Bare addon in C++ and JS runtime.
class BertInterface {
  /**
   *
   * @param {Object} configurationParams - all the required configuration for inference setup
   * @param {Function} outputCb - to be called on any inference event ( started, new output, error, etc )
   */
  constructor (binding, configurationParams, outputCb) {
    this._binding = binding

    if (!configurationParams.backendsDir) {
      configurationParams.backendsDir = path.join(__dirname, 'prebuilds')
    }

    this._handle = binding.createInstance(this, configurationParams, outputCb)
  } ///

  /**
   * Cancel current inference process. Resolves when the job has stopped.
   */
  async cancel () {
    if (!this._handle) return
    await this._binding.cancel(this._handle)
  }

  /**
   * Processes new input
   * @param {Object} data
   * @param {String} data.type - Either 'text' for string input or 'sequences' for string array input
   * @param {String|Array<String>} data.input - Input text (for 'text') or array of texts (for 'sequences')
   * @returns {Promise<bool>} true if the job was accepted, false if busy
   */
  async runJob (data) {
    return this._binding.runJob(this._handle, data)
  }

  /**
   * Loads model weights. The native side reads the JS property names
   * `chunk` and `completed` directly, so this object's field names are
   * load-bearing — see `JsBlobsStream.hpp::appendBlob` in
   * `qvac-lib-inference-addon-cpp` for the parser.
   * @param {Object} data
   * @param {String} data.filename - Logical filename used to group chunks
   *   into one shard. The native side keys `shards_in_progress` on this.
   * @param {Uint8Array|null} data.chunk - Next chunk of bytes for the
   *   current shard, or `null` on the final call when `completed` is true.
   * @param {Boolean} data.completed - `false` while more chunks remain;
   *   `true` on the last call to finalize the shard.
   */
  async loadWeights (data) {
    return this._binding.loadWeights(this._handle, data)
  }

  /**
   * Activates the model to start processing the queue
   */
  async activate () {
    return this._binding.activate(this._handle)
  }

  /**
   * Stops addon process and clears resources (including memory).
   */
  async unload () {
    if (!this._handle) return
    this._binding.destroyInstance(this._handle)
    this._handle = null
  }
}

module.exports = {
  BertInterface,
  mapAddonEvent
}
