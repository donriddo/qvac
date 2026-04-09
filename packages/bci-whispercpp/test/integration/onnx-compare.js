'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const { spawnSync } = require('bare-subprocess')

const fixturesDir = path.join(__dirname, '..', 'fixtures')
const manifest = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'manifest.json'), 'utf8'))
const pythonPreds = JSON.parse(fs.readFileSync(path.join(fixturesDir, 'python_predictions.json'), 'utf8'))

const MODELS_DIR = path.join(__dirname, '..', '..', 'models', 'onnx')
const CHECKPOINT = '/Users/rajusharma/Downloads/brainwhisperer-qvac/epoch=93-val_wer=0.0910.ckpt'
const ARGS_PATH = '/Users/rajusharma/Downloads/brainwhisperer-qvac/rnn_args.yaml'
const MODEL_DIR = '/Users/rajusharma/Downloads/brainwhisperer-qvac'
const SCRIPT = path.join(__dirname, '..', '..', 'scripts', 'onnx-infer.py')

function computeWER (hypothesis, reference) {
  const hyp = hypothesis.toLowerCase().trim().split(/\s+/).filter(Boolean)
  const ref = reference.toLowerCase().trim().split(/\s+/).filter(Boolean)
  if (ref.length === 0) return hyp.length === 0 ? 0 : 1
  const n = ref.length; const m = hyp.length
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0))
  for (let i = 0; i <= n; i++) dp[i][0] = i
  for (let j = 0; j <= m; j++) dp[0][j] = j
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (ref[i - 1] === hyp[j - 1]) dp[i][j] = dp[i - 1][j - 1]
      else dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    }
  }
  return dp[n][m] / n
}

const hasOnnx = fs.existsSync(path.join(MODELS_DIR, 'bci_encoder.onnx')) &&
                fs.existsSync(path.join(MODELS_DIR, 'bci_decoder.onnx'))
const hasCheckpoint = fs.existsSync(CHECKPOINT)

if (!hasOnnx || !hasCheckpoint) {
  console.log('SKIP: ONNX models or checkpoint not found')
  process.exit(0)
}

console.log('='.repeat(60))
console.log('ONNX Inference vs Python Predictions')
console.log('='.repeat(60))

let totalWer = 0
let matchCount = 0

for (let i = 0; i < manifest.samples.length; i++) {
  const sample = manifest.samples[i]
  const samplePath = path.join(fixturesDir, sample.file)

  const spawnResult = spawnSync('python3', [
    SCRIPT,
    '--signal', samplePath,
    '--models-dir', MODELS_DIR,
    '--checkpoint', CHECKPOINT,
    '--args', ARGS_PATH,
    '--model-dir', MODEL_DIR,
    '--day-idx', String(sample.day_idx || 1)
  ], { timeout: 120000 })

  if (spawnResult.status !== 0) {
    console.log(`  ERROR: ${Buffer.from(spawnResult.stderr).toString()}`)
    continue
  }
  const stdout = Buffer.from(spawnResult.stdout).toString()
  const lines = stdout.trim().split('\n')
  const jsonLine = lines[lines.length - 1]
  const result = JSON.parse(jsonLine)
  const onnxText = result.text

  const pyPred = pythonPreds[i] ? pythonPreds[i].prediction : 'N/A'
  const werVsExpected = computeWER(onnxText, sample.expected_text)
  const werVsPython = computeWER(onnxText, pyPred)
  const matchesPython = onnxText === pyPred

  totalWer += werVsExpected
  if (matchesPython) matchCount++

  console.log(`\n  Sample ${i}: ${sample.file}`)
  console.log(`    Expected:   "${sample.expected_text}"`)
  console.log(`    Python:     "${pyPred}"`)
  console.log(`    ONNX:       "${onnxText}"`)
  console.log(`    Match py:   ${matchesPython ? 'YES' : 'NO'}`)
  console.log(`    WER vs exp: ${(werVsExpected * 100).toFixed(1)}%`)
}

const avgWer = totalWer / manifest.samples.length
console.log(`\n${'='.repeat(60)}`)
console.log(`  Average WER vs expected: ${(avgWer * 100).toFixed(1)}%`)
console.log(`  Python match: ${matchCount}/${manifest.samples.length}`)
console.log(`${'='.repeat(60)}`)

if (matchCount === manifest.samples.length) {
  console.log('\nSUCCESS: All ONNX predictions match Python beam search!')
} else {
  console.log(`\nWARNING: ${manifest.samples.length - matchCount} samples differ from Python`)
}
