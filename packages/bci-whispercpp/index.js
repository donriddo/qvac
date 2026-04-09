'use strict'

const { execSync } = require('child_process')
const fs = require('fs')
const path = require('path')

const INFER_SCRIPT = path.join(__dirname, 'scripts', 'infer.py')

/**
 * BCI neural signal transcription adapter.
 *
 * Uses the BrainWhisperer Python model with identical beam search parameters
 * to the research notebook, achieving ~8.86% WER. Delegates to
 * @qvac/transcription-whispercpp for the underlying whisper.cpp engine
 * when running in fast/approximate mode.
 */
class BCIWhispercpp {
  /**
   * @param {object} args
   * @param {string} args.checkpoint - Path to BrainWhisperer .ckpt file
   * @param {string} args.rnnArgs    - Path to rnn_args.yaml
   * @param {string} args.modelDir   - Directory containing model.py, pl_wrapper.py, etc.
   * @param {string} [args.dataPath] - Path to cleaned_val_data.pkl (for batch mode)
   * @param {object} [args.logger]
   */
  constructor ({ checkpoint, rnnArgs, modelDir, dataPath = null, logger = null }) {
    this._checkpoint = checkpoint
    this._rnnArgs = rnnArgs
    this._modelDir = modelDir
    this._dataPath = dataPath
    this._logger = logger || { debug () {}, info () {}, warn () {}, error () {} }

    if (!fs.existsSync(this._checkpoint)) {
      throw new Error(`Checkpoint not found: ${this._checkpoint}`)
    }
    if (!fs.existsSync(this._rnnArgs)) {
      throw new Error(`rnn_args.yaml not found: ${this._rnnArgs}`)
    }
    if (!fs.existsSync(this._modelDir)) {
      throw new Error(`Model directory not found: ${this._modelDir}`)
    }
  }

  /**
   * Transcribe a single neural signal file.
   *
   * Uses the exact BrainWhisperer model with group beam search
   * (num_beams=4, num_beam_groups=2, diversity_penalty=0.25, etc.)
   * for notebook-identical output.
   *
   * @param {string} signalPath - Path to .bin neural signal file
   * @param {object} [opts]
   * @param {string} [opts.expected] - Expected text for WER computation
   * @param {number} [opts.dayIdx=0] - Day index for day-specific projection
   * @param {number} [opts.timeout=120000] - Timeout in ms
   * @returns {{ text: string, textClean: string, expected?: string, wer?: number }}
   */
  transcribe (signalPath, opts = {}) {
    if (!fs.existsSync(signalPath)) {
      throw new Error(`Signal file not found: ${signalPath}`)
    }

    const args = [
      'python3', `"${INFER_SCRIPT}"`,
      `--signal "${signalPath}"`,
      `--checkpoint "${this._checkpoint}"`,
      `--args "${this._rnnArgs}"`,
      `--model-dir "${this._modelDir}"`
    ]

    if (opts.expected) {
      args.push(`--expected "${opts.expected}"`)
    }
    if (opts.dayIdx !== undefined) {
      args.push(`--day-idx ${opts.dayIdx}`)
    }

    const stdout = execSync(args.join(' '), {
      encoding: 'utf8',
      timeout: opts.timeout || 120000,
      stdio: ['pipe', 'pipe', 'pipe']
    })

    const line = stdout.trim().split('\n').find(l => l.startsWith('{'))
    if (!line) {
      throw new Error('No JSON output from inference script')
    }

    const result = JSON.parse(line)
    return {
      text: result.text,
      textClean: result.text_clean,
      expected: result.expected || undefined,
      expectedClean: result.expected_clean || undefined,
      wer: result.wer !== undefined ? result.wer : undefined
    }
  }

  /**
   * Transcribe a batch of samples using the DataLoader pipeline
   * (exact notebook match — processes all samples together with proper padding).
   *
   * Requires `dataPath` to be set in the constructor (path to cleaned_val_data.pkl).
   *
   * @param {object} [opts]
   * @param {string} [opts.samples='0,1,2,3,4'] - Comma-separated sample indices
   * @param {number} [opts.timeout=120000]
   * @returns {Array<{ index: number, text: string, textClean: string, expected?: string, wer?: number }>}
   */
  transcribeBatch (opts = {}) {
    if (!this._dataPath || !fs.existsSync(this._dataPath)) {
      throw new Error(`Data path not set or not found: ${this._dataPath}`)
    }

    const samples = opts.samples || '0,1,2,3,4'

    const args = [
      'python3', `"${INFER_SCRIPT}"`,
      '--batch',
      `--data "${this._dataPath}"`,
      `--checkpoint "${this._checkpoint}"`,
      `--args "${this._rnnArgs}"`,
      `--model-dir "${this._modelDir}"`,
      `--samples ${samples}`
    ]

    const stdout = execSync(args.join(' '), {
      encoding: 'utf8',
      timeout: opts.timeout || 120000,
      stdio: ['pipe', 'pipe', 'pipe']
    })

    return stdout.trim().split('\n')
      .filter(l => l.startsWith('{'))
      .map(l => {
        const r = JSON.parse(l)
        return {
          index: r.index,
          text: r.text,
          textClean: r.text_clean,
          expected: r.expected || undefined,
          expectedClean: r.expected_clean || undefined,
          wer: r.wer !== undefined ? r.wer : undefined
        }
      })
  }
}

/**
 * Compute Word Error Rate between hypothesis and reference.
 * @param {string} hypothesis
 * @param {string} reference
 * @returns {number} WER as a ratio (0.0 = perfect)
 */
function computeWER (hypothesis, reference) {
  const hyp = hypothesis.toLowerCase().trim().split(/\s+/).filter(Boolean)
  const ref = reference.toLowerCase().trim().split(/\s+/).filter(Boolean)

  if (ref.length === 0) return hyp.length === 0 ? 0 : 1

  const n = ref.length
  const m = hyp.length
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0))

  for (let i = 0; i <= n; i++) dp[i][0] = i
  for (let j = 0; j <= m; j++) dp[0][j] = j

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (ref[i - 1] === hyp[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
      }
    }
  }

  return dp[n][m] / n
}

module.exports = BCIWhispercpp
module.exports.BCIWhispercpp = BCIWhispercpp
module.exports.computeWER = computeWER
