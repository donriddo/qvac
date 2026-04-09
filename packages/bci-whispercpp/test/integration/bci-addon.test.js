'use strict'

const fs = require('fs')
const path = require('path')
const { BCIWhispercpp, computeWER } = require('../..')

const BRAINWHISPERER_DIR = path.join(
  process.env.HOME || '', 'Downloads', 'brainwhisperer-qvac'
)

const CHECKPOINT = path.join(BRAINWHISPERER_DIR, 'epoch=93-val_wer=0.0910.ckpt')
const RNN_ARGS = path.join(BRAINWHISPERER_DIR, 'rnn_args.yaml')
const DATA_PATH = path.join(BRAINWHISPERER_DIR, 'cleaned_val_data.pkl')
const FIXTURES = path.join(__dirname, '..', 'fixtures')

const hasModel = fs.existsSync(CHECKPOINT) && fs.existsSync(RNN_ARGS)

function assert (condition, message) {
  if (!condition) {
    console.error(`FAIL: ${message}`)
    process.exit(1)
  }
  console.log(`  PASS: ${message}`)
}

function test (name, fn) {
  console.log(`\n# ${name}`)
  try {
    fn()
    console.log(`ok - ${name}`)
  } catch (err) {
    console.error(`not ok - ${name}: ${err.message}`)
    process.exit(1)
  }
}

if (!hasModel) {
  console.log('Skipping tests: BrainWhisperer model not found at', BRAINWHISPERER_DIR)
  process.exit(0)
}

const bci = new BCIWhispercpp({
  checkpoint: CHECKPOINT,
  rnnArgs: RNN_ARGS,
  modelDir: BRAINWHISPERER_DIR,
  dataPath: DATA_PATH
})

test('single file transcription', () => {
  const signalPath = path.join(FIXTURES, 'neural_sample_2.bin')
  if (!fs.existsSync(signalPath)) {
    console.log('  SKIP: fixture not found')
    return
  }
  const result = bci.transcribe(signalPath, { expected: 'Not too controversial.' })

  assert(typeof result.text === 'string', 'should return text')
  assert(result.text.length > 0, 'text should be non-empty')
  assert(result.wer !== undefined, 'should compute WER')
  console.log(`  Text: "${result.text}", WER: ${(result.wer * 100).toFixed(1)}%`)
})

test('batch transcription matches notebook', () => {
  const results = bci.transcribeBatch()

  assert(results.length === 5, 'should return 5 results')

  const expectedPredictions = [
    'You can see the good at this point as well.',
    'How does it keep the cost said?',
    'Not too controversial.',
    'The jury and a judge work together on it.',
    "We're quite vocal about it."
  ]

  let totalWer = 0
  for (let i = 0; i < results.length; i++) {
    const r = results[i]
    assert(r.text === expectedPredictions[i],
      `sample ${i}: "${r.text}" === "${expectedPredictions[i]}"`)
    if (r.wer !== undefined) totalWer += r.wer
  }

  const avgWer = totalWer / results.length
  console.log(`\n  Average WER: ${(avgWer * 100).toFixed(2)}%`)
  assert(avgWer < 0.12, `average WER ${(avgWer * 100).toFixed(1)}% should be < 12%`)
})

test('computeWER function', () => {
  assert(computeWER('hello world', 'hello world') === 0, 'identical = 0')
  assert(computeWER('hello', 'hello world') === 0.5, 'deletion = 0.5')
  assert(computeWER('hello world foo', 'hello world') === 0.5, 'insertion = 0.5')
  assert(computeWER('goodbye world', 'hello world') === 0.5, 'substitution = 0.5')
})

console.log('\n# all tests passed')
