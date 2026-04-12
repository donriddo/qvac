'use strict'

const path = require('bare-path')

/**
 * Map a raw native event from the C++ stable-diffusion addon to a logical
 * event consumed by `ImgStableDiffusion`.
 *
 * The native binding emits events with C++-mangled names and varied
 * payload shapes. This wrapper normalizes them into one of:
 *   - `'Output'`     — image bytes (`Uint8Array`) or progress JSON tick (`string`)
 *   - `'Error'`      — failure
 *   - `'JobEnded'`   — terminal RuntimeStats payload (object)
 *
 * Returns `{ type, data, error }` or `null` for unknown event/data shapes
 * (caller logs at debug level).
 *
 *
 * @param {string} rawEvent
 * @param {*} rawData
 * @param {*} rawError
 * @returns {{ type: string, data: *, error: * } | null}
 */
function mapAddonEvent (rawEvent, rawData, rawError) {
  if (typeof rawEvent === 'string' && rawEvent.includes('Error')) {
    return { type: 'Error', data: rawData, error: rawError }
  }

  if (rawData instanceof Uint8Array || typeof rawData === 'string') {
    return { type: 'Output', data: rawData, error: null }
  }

  if (rawData && typeof rawData === 'object') {
    return { type: 'JobEnded', data: rawData, error: null }
  }

  return null
}

/**
 * JavaScript wrapper around the native stable-diffusion.cpp addon.
 * Manages the native handle lifecycle and bridges JS ↔ C++.
 */
class SdInterface {
  /**
   * @param {object} binding - The native addon binding (from require.addon())
   * @param {object} configurationParams - Configuration for the SD context
   * @param {string} configurationParams.path - Local file path to the model weights
   * @param {object} [configurationParams.config] - SD-specific configuration options
   * @param {Function} outputCb - Called on any generation event (started, progress, output, error)
   */
  constructor (binding, configurationParams, outputCb) {
    this._binding = binding

    if (!configurationParams.config) {
      configurationParams.config = {}
    }

    if (!configurationParams.config.backendsDir) {
      configurationParams.config.backendsDir = path.join(__dirname, 'prebuilds')
    }

    // C++ getSubmap expects every config value to be a JS string.
    // Coerce numbers and booleans here so the native layer never sees non-string values.
    configurationParams.config = Object.fromEntries(
      Object.entries(configurationParams.config)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    )

    this._handle = this._binding.createInstance(
      this,
      configurationParams,
      outputCb
    )
  }

  /**
   * Moves addon to the LISTENING state after initialization.
   */
  async activate () {
    this._binding.activate(this._handle)
  }

  /**
   * Cancel the current generation job.
   */
  async cancel () {
    if (!this._handle) return
    await this._binding.cancel(this._handle)
  }

  /**
   * Run a generation job with the given parameters.
   * @param {object} params - Generation parameters (will be JSON-serialized)
   * @returns {Promise<boolean>} true if job was accepted, false if busy
   */
  async runJob (params) {
    const paramsJson = JSON.stringify(params)
    return this._binding.runJob(this._handle, { type: 'text', input: paramsJson })
  }

  /**
   * Destroy the native instance and release all resources.
   * After this the SdInterface object must not be used.
   */
  async unload () {
    if (!this._handle) return
    this._binding.destroyInstance(this._handle)
    this._handle = null
  }
}

module.exports = { SdInterface, mapAddonEvent }
