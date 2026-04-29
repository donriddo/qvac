'use strict'

const test = require('brittle')
const path = require('bare-path')
const LlmLlamacpp = require('../../index.js')
const { ensureModel } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')

const os = require('bare-os')
const isAndroid = os.platform() === 'android'

const BITNET_MODEL = {
  name: 'bitnet_b1_58-large-TQ2_0.gguf',
  url: 'https://huggingface.co/gianni-cor/bitnet_b1_58-large-TQ2_0/resolve/main/bitnet_b1_58-large-TQ2_0.gguf'
}

const PROMPT = [
  { role: 'user', content: 'What is 2 + 2?' }
]

async function collectResponse (response) {
  const chunks = []
  await response
    .onUpdate(data => { chunks.push(data) })
    .await()
  return chunks.join('').trim()
}

test('bitnet model can run simple inference', { timeout: 600_000, skip: !isAndroid }, async t => {
  let addon = null
  const specLogger = attachSpecLogger({ forwardToConsole: true })
  try {
    const [modelName, dirPath] = await ensureModel({
      modelName: BITNET_MODEL.name,
      downloadUrl: BITNET_MODEL.url
    })

    const modelPath = path.join(dirPath, modelName)

    const config = {
      gpu_layers: '999',
      ctx_size: '1024',
      device: 'gpu',
      n_predict: '32',
      verbosity: '2'
    }

    addon = new LlmLlamacpp({
      files: { model: [modelPath] },
      config,
      logger: console,
      opts: { stats: true }
    })

    await addon.load()
    const response = await addon.run(PROMPT)
    const output = await collectResponse(response)

    t.ok(output.length > 0, 'bitnet model should generate output')
    t.comment(`BitNet output: "${output}"`)
  } catch (error) {
    console.error(error)
    t.fail('bitnet model can run simple inference: ' + error.message)
  } finally {
    if (addon) await addon.unload().catch(() => {})
    specLogger.release()
  }
})
