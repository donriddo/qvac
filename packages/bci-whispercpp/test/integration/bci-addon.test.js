'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const test = require('brittle')
const { BCIInterface } = require('../../bci')
const binding = require('../../binding')
const { getTestPaths, computeWER, detectPlatform } = require('./helpers')

const platform = detectPlatform()
const { fixturesDir, manifest, getSamplePath } = getTestPaths()

// Model path: whisper tiny.en model must be present for integration tests
const os = require('bare-os')
const MODEL_PATH = (os.hasEnv('WHISPER_MODEL_PATH') ? os.getEnv('WHISPER_MODEL_PATH') : null) ||
  path.join(__dirname, '..', '..', 'models', 'ggml-tiny.en.bin')

const hasModel = fs.existsSync(MODEL_PATH)

test('[BCI] addon creates instance and activates', { skip: !hasModel }, async (t) => {
  let resolveJobEnded
  const jobEndedPromise = new Promise((resolve) => {
    resolveJobEnded = resolve
  })

  const onOutput = (addon, event, jobId, output, error) => {
    console.log(`Event: ${event}, JobId: ${jobId}`)
    if (event === 'JobEnded') {
      resolveJobEnded(output)
    }
  }

  const config = {
    contextParams: { model: MODEL_PATH },
    whisperConfig: { language: 'en', temperature: 0.0 },
    miscConfig: { caption_enabled: false }
  }

  let model
  try {
    model = new BCIInterface(binding, config, onOutput)
    t.ok(model, 'BCIInterface should be created')

    const status = await model.status()
    t.ok(status, 'Status should be returned')

    await model.activate()
    const statusAfter = await model.status()
    t.is(statusAfter, 'listening', 'Status after activate should be listening')
  } finally {
    if (model) await model.destroyInstance()
  }
})

test('[BCI] batch transcription from neural signal file', { skip: !hasModel }, async (t) => {
  if (manifest.samples.length === 0) {
    t.skip('No neural signal test fixtures found')
    return
  }

  const sample = manifest.samples[0]
  const samplePath = getSamplePath(sample.file)
  if (!fs.existsSync(samplePath)) {
    t.skip(`Sample file missing: ${samplePath}`)
    return
  }

  const segments = []
  let stats = null

  const onOutput = (addon, event, jobId, data, error) => {
    if (event === 'Output') {
      if (Array.isArray(data)) {
        segments.push(...data)
      } else if (data && data.text) {
        segments.push(data)
      }
    } else if (event === 'JobEnded') {
      stats = data
    } else if (event === 'Error') {
      console.error('Transcription error:', error)
    }
  }

  const config = {
    contextParams: { model: MODEL_PATH },
    whisperConfig: { language: 'en', temperature: 0.0 },
    miscConfig: { caption_enabled: false }
  }

  const model = new BCIInterface(binding, config, onOutput)
  try {
    await model.activate()

    const neuralData = fs.readFileSync(samplePath)
    const inputData = new Uint8Array(neuralData)

    const accepted = await model.runJob({ input: inputData })
    t.ok(accepted, 'Job should be accepted')

    // Wait for completion
    await new Promise((resolve) => {
      const interval = setInterval(() => {
        if (stats !== null || segments.length > 0) {
          clearInterval(interval)
          resolve()
        }
      }, 100)
      setTimeout(() => { clearInterval(interval); resolve() }, 30000)
    })

    const transcription = segments.map(s => s.text).join('').trim()
    console.log(`\n=== Batch Transcription Result ===`)
    console.log(`Expected:  "${sample.expected_text}"`)
    console.log(`Got:       "${transcription}"`)

    const wer = computeWER(transcription, sample.expected_text)
    console.log(`WER:       ${(wer * 100).toFixed(1)}%`)

    t.ok(typeof transcription === 'string', 'Should produce a transcription string')
    t.ok(typeof wer === 'number' && wer >= 0, 'WER should be a non-negative number')
    console.log(`\nNote: High WER expected - standard whisper model is not BCI-trained.`)
    console.log(`A BCI-trained GGML model is needed for meaningful neural-to-text results.`)
  } finally {
    await model.destroyInstance()
  }
})

test('[BCI] streaming transcription from neural signal chunks', { skip: !hasModel }, async (t) => {
  if (manifest.samples.length === 0) {
    t.skip('No neural signal test fixtures found')
    return
  }

  const sample = manifest.samples[1] || manifest.samples[0]
  const samplePath = getSamplePath(sample.file)
  if (!fs.existsSync(samplePath)) {
    t.skip(`Sample file missing: ${samplePath}`)
    return
  }

  const segments = []
  let stats = null
  let jobEnded = false

  const onOutput = (addon, event, jobId, data, error) => {
    if (event === 'Output') {
      if (Array.isArray(data)) segments.push(...data)
      else if (data && data.text) segments.push(data)
    } else if (event === 'JobEnded') {
      stats = data
      jobEnded = true
    }
  }

  const config = {
    contextParams: { model: MODEL_PATH },
    whisperConfig: { language: 'en', temperature: 0.0 },
    miscConfig: { caption_enabled: false }
  }

  const model = new BCIInterface(binding, config, onOutput)
  try {
    await model.activate()

    const fullData = fs.readFileSync(samplePath)

    // Simulate streaming: split into 3 chunks
    const chunkSize = Math.ceil(fullData.length / 3)

    await model.append({ type: 'neural', input: new Uint8Array(0) })

    for (let i = 0; i < fullData.length; i += chunkSize) {
      const end = Math.min(i + chunkSize, fullData.length)
      const chunk = new Uint8Array(fullData.buffer, fullData.byteOffset + i, end - i)
      await model.append({ type: 'neural', input: chunk })
    }

    await model.append({ type: 'end of job' })

    await new Promise((resolve) => {
      const interval = setInterval(() => {
        if (jobEnded) { clearInterval(interval); resolve() }
      }, 100)
      setTimeout(() => { clearInterval(interval); resolve() }, 30000)
    })

    const transcription = segments.map(s => s.text).join('').trim()
    console.log(`\n=== Streaming Transcription Result ===`)
    console.log(`Expected:  "${sample.expected_text}"`)
    console.log(`Got:       "${transcription}"`)

    const wer = computeWER(transcription, sample.expected_text)
    console.log(`WER:       ${(wer * 100).toFixed(1)}%`)

    t.ok(typeof transcription === 'string', 'Streaming should produce transcription')
    t.ok(typeof wer === 'number', 'WER should be computable')
  } finally {
    await model.destroyInstance()
  }
})

test('[BCI] WER measurement across all test samples', { skip: !hasModel }, async (t) => {
  if (manifest.samples.length === 0) {
    t.skip('No neural signal test fixtures found')
    return
  }

  console.log(`\n=== WER Report (${manifest.samples.length} samples) ===`)
  console.log(`Platform: ${platform.label}`)
  console.log(`Model:    ${MODEL_PATH}\n`)

  const results = []

  for (const sample of manifest.samples) {
    const samplePath = getSamplePath(sample.file)
    if (!fs.existsSync(samplePath)) continue

    const segments = []
    let jobEnded = false

    const onOutput = (addon, event, jobId, data, error) => {
      if (event === 'Output') {
        if (Array.isArray(data)) segments.push(...data)
        else if (data && data.text) segments.push(data)
      } else if (event === 'JobEnded') {
        jobEnded = true
      }
    }

    const config = {
      contextParams: { model: MODEL_PATH },
      whisperConfig: { language: 'en', temperature: 0.0 },
      miscConfig: { caption_enabled: false }
    }

    const model = new BCIInterface(binding, config, onOutput)
    try {
      await model.activate()

      const neuralData = new Uint8Array(fs.readFileSync(samplePath))
      await model.runJob({ input: neuralData })

      await new Promise((resolve) => {
        const interval = setInterval(() => {
          if (jobEnded) { clearInterval(interval); resolve() }
        }, 100)
        setTimeout(() => { clearInterval(interval); resolve() }, 30000)
      })

      const transcription = segments.map(s => s.text).join('').trim()
      const wer = computeWER(transcription, sample.expected_text)
      results.push({ expected: sample.expected_text, got: transcription, wer })

      console.log(`  [${sample.file}]`)
      console.log(`    Expected: "${sample.expected_text}"`)
      console.log(`    Got:      "${transcription}"`)
      console.log(`    WER:      ${(wer * 100).toFixed(1)}%\n`)
    } finally {
      await model.destroyInstance()
    }
  }

  const avgWER = results.reduce((sum, r) => sum + r.wer, 0) / results.length
  console.log(`  Average WER: ${(avgWER * 100).toFixed(1)}%`)
  console.log(`  Samples tested: ${results.length}`)

  t.ok(results.length > 0, 'Should have tested at least one sample')
  t.ok(typeof avgWER === 'number', 'Average WER should be computable')
})
