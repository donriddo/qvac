'use strict'

/**
 * Transcribe neural signal files using the BCI BrainWhisperer model.
 * Uses the Python inference backend for exact notebook-matching output.
 *
 * Usage:
 *   node examples/transcribe-neural.js <signal.bin> [checkpoint] [rnn_args.yaml] [model_dir]
 *
 * Or batch mode (matches notebook exactly):
 *   node examples/transcribe-neural.js --batch [data.pkl] [checkpoint] [rnn_args.yaml] [model_dir]
 */

const { execSync } = require('child_process')
const fs = require('fs')
const path = require('path')

const BRAINWHISPERER_DIR = path.join(
  process.env.HOME || '', 'Downloads', 'brainwhisperer-qvac'
)
const DEFAULT_CHECKPOINT = path.join(BRAINWHISPERER_DIR, 'epoch=93-val_wer=0.0910.ckpt')
const DEFAULT_ARGS = path.join(BRAINWHISPERER_DIR, 'rnn_args.yaml')
const DEFAULT_DATA = path.join(BRAINWHISPERER_DIR, 'cleaned_val_data.pkl')

function main () {
  const args = process.argv.slice(2)
  const isBatch = args[0] === '--batch'

  if (args.length < 1) {
    console.log('Usage:')
    console.log('  Single: node examples/transcribe-neural.js <signal.bin>')
    console.log('  Batch:  node examples/transcribe-neural.js --batch')
    return
  }

  const inferScript = path.join(__dirname, '..', 'scripts', 'infer.py')
  const checkpoint = (isBatch ? args[2] : args[1]) || DEFAULT_CHECKPOINT
  const rnnArgs = (isBatch ? args[3] : args[2]) || DEFAULT_ARGS
  const modelDir = (isBatch ? args[4] : args[3]) || BRAINWHISPERER_DIR

  if (isBatch) {
    const dataPath = args[1] || DEFAULT_DATA
    console.log('=== BCI Neural Signal Transcription (Batch Mode) ===')
    console.log(`Data:       ${dataPath}`)
    console.log(`Checkpoint: ${checkpoint}`)
    console.log('')

    const startTime = Date.now()
    const stdout = execSync(
      `python3 "${inferScript}" --batch ` +
      `--data "${dataPath}" ` +
      `--checkpoint "${checkpoint}" ` +
      `--args "${rnnArgs}" ` +
      `--model-dir "${modelDir}" ` +
      '--samples 0,1,2,3,4',
      { encoding: 'utf8', timeout: 120000, stdio: ['pipe', 'pipe', 'pipe'] }
    )

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)
    const results = stdout.trim().split('\n').filter(l => l.startsWith('{')).map(l => JSON.parse(l))

    let totalWer = 0
    for (const r of results) {
      console.log(`Sample ${r.index}:`)
      console.log(`  Got:      "${r.text}"`)
      if (r.expected) {
        console.log(`  Expected: "${r.expected}"`)
        console.log(`  WER:      ${(r.wer * 100).toFixed(1)}%`)
        totalWer += r.wer
      }
      console.log('')
    }

    const avgWer = totalWer / results.length
    console.log(`Average WER: ${(avgWer * 100).toFixed(2)}%`)
    console.log(`Time: ${elapsed}s`)
  } else {
    const signalPath = args[0]
    if (!fs.existsSync(signalPath)) {
      console.error(`Error: Signal file not found: ${signalPath}`)
      process.exit(1)
    }

    const buf = fs.readFileSync(signalPath)
    const T = buf.readUInt32LE(0)
    const C = buf.readUInt32LE(4)

    console.log('=== BCI Neural Signal Transcription ===')
    console.log(`Signal:     ${signalPath}`)
    console.log(`Timesteps:  ${T}, Channels: ${C}`)
    console.log(`Duration:   ~${(T * 20 / 1000).toFixed(1)}s`)
    console.log('')

    const startTime = Date.now()
    const stdout = execSync(
      `python3 "${inferScript}" ` +
      `--signal "${signalPath}" ` +
      `--checkpoint "${checkpoint}" ` +
      `--args "${rnnArgs}" ` +
      `--model-dir "${modelDir}"`,
      { encoding: 'utf8', timeout: 120000, stdio: ['pipe', 'pipe', 'pipe'] }
    )

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)
    const line = stdout.trim().split('\n').find(l => l.startsWith('{'))
    const result = JSON.parse(line)

    console.log(`Text: "${result.text}"`)
    console.log(`Time: ${elapsed}s`)
  }

  console.log('\nDone.')
}

main()
