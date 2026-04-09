'use strict'

/**
 * Transcribe neural signal files using the BCI BrainWhisperer model.
 *
 * Usage:
 *   node examples/transcribe-neural.js <signal.bin>
 *   node examples/transcribe-neural.js --batch
 */

const fs = require('fs')
const path = require('path')
const { BCIWhispercpp, computeWER } = require('..')

const BRAINWHISPERER_DIR = path.join(
  process.env.HOME || '', 'Downloads', 'brainwhisperer-qvac'
)

function main () {
  const args = process.argv.slice(2)

  if (args.length < 1) {
    console.log('Usage:')
    console.log('  Single: node examples/transcribe-neural.js <signal.bin>')
    console.log('  Batch:  node examples/transcribe-neural.js --batch')
    return
  }

  const bci = new BCIWhispercpp({
    checkpoint: path.join(BRAINWHISPERER_DIR, 'epoch=93-val_wer=0.0910.ckpt'),
    rnnArgs: path.join(BRAINWHISPERER_DIR, 'rnn_args.yaml'),
    modelDir: BRAINWHISPERER_DIR,
    dataPath: path.join(BRAINWHISPERER_DIR, 'cleaned_val_data.pkl')
  })

  if (args[0] === '--batch') {
    console.log('=== BCI Neural Signal Transcription (Batch) ===\n')

    const startTime = Date.now()
    const results = bci.transcribeBatch()
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)

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

    console.log(`Average WER: ${((totalWer / results.length) * 100).toFixed(2)}%`)
    console.log(`Time: ${elapsed}s\nDone.`)
  } else {
    const signalPath = args[0]
    if (!fs.existsSync(signalPath)) {
      console.error(`Error: File not found: ${signalPath}`)
      process.exit(1)
    }

    const buf = fs.readFileSync(signalPath)
    const T = buf.readUInt32LE(0)
    const C = buf.readUInt32LE(4)

    console.log('=== BCI Neural Signal Transcription ===')
    console.log(`Signal:    ${signalPath}`)
    console.log(`Shape:     ${T} timesteps x ${C} channels (~${(T * 20 / 1000).toFixed(1)}s)\n`)

    const startTime = Date.now()
    const result = bci.transcribe(signalPath)
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)

    console.log(`Text: "${result.text}"`)
    console.log(`Time: ${elapsed}s\nDone.`)
  }
}

main()
