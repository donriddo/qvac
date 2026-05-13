'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const subprocess = require('bare-subprocess')
const ImgStableDiffusion = require('../index')

/**
 * FLUX2-klein F16 img2img example
 *
 * Full precision (F16) version for comparison with Q8_0 quantized model.
 * This should have much less quantization bias.
 */

function makeBar (step, total, width) {
  const filled = Math.round((step / total) * width)
  const complete = step >= total
  const inner = complete
    ? '='.repeat(width)
    : '='.repeat(Math.max(0, filled - 1)) + '>' + ' '.repeat(Math.max(0, width - filled))
  return '[' + inner + ']'
}

async function gpuMemory () {
  return new Promise((resolve) => {
    let out = ''
    let proc
    try {
      proc = subprocess.spawn('nvidia-smi', [
        '--query-gpu=index,memory.used,memory.total',
        '--format=csv,noheader,nounits'
      ], { stdio: ['ignore', 'pipe', 'ignore'] })
    } catch (_) {
      resolve([])
      return
    }
    proc.stdout.on('data', (chunk) => { out += chunk.toString() })
    proc.on('close', () => {
      const gpus = out.trim().split('\n').filter(Boolean).map((line) => {
        const parts = line.split(',').map(s => parseInt(s.trim(), 10))
        return { idx: parts[0], used: parts[1], total: parts[2] }
      })
      resolve(gpus)
    })
    proc.on('error', () => resolve([]))
  })
}

function fmtGpus (gpus) {
  return gpus.map(g => `GPU${g.idx}: ${g.used}/${g.total} MB`).join('  ')
}

async function main () {
  const modelDir = path.join(__dirname, '../models')
  const inputImagePath = path.join(__dirname, '../assets/von-neumann.jpg')
  const outputImagePath = path.join(__dirname, '../temp/von-neumann_transformed_f16.png')

  if (!fs.existsSync(inputImagePath)) {
    console.error(`Error: Input image not found at ${inputImagePath}`)
    return
  }

  const outputDir = path.dirname(outputImagePath)
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true })

  console.log('Loading FLUX2-klein F16 model (full precision)...')

  const model = new ImgStableDiffusion({
    files: {
      model: path.join(modelDir, 'flux-2-klein-4b-F16.gguf'), // F16 full precision
      llm: path.join(modelDir, 'Qwen3-4B-Q8_0.gguf'), // Q8 text encoder
      vae: path.join(modelDir, 'flux2-vae.safetensors')
    },
    config: {
      threads: 4,
      device: 'gpu',
      prediction: 'flux2_flow',
      diffusion_fa: true
    },
    logger: console
  })

  try {
    await model.load()
    console.log('Model loaded!')

    const initImage = fs.readFileSync(inputImagePath)
    console.log(`Input image: ${initImage.length} bytes`)

    const gpusPre = await gpuMemory()

    const STEPS = 20
    const STRENGTH = 1.0
    const GUIDANCE = 9.0
    const SEED = -1

    console.log('\n=== F16 Full Precision Model ===')
    console.log('  Model    : flux-2-klein-4b-F16.gguf (16-bit)')
    console.log('  Steps    : ' + STEPS)
    console.log('  Strength : ' + STRENGTH)
    console.log('  Guidance : ' + GUIDANCE)
    console.log('  Seed     : ' + SEED)
    if (gpusPre.length) console.log('  GPU pre  : ' + fmtGpus(gpusPre))
    console.log('  Note     : VAE encode runs first (no progress tick) — please wait...\n')

    const tGenStart = Date.now()
    let lastStepTime = tGenStart

    const response = await model.run({
      prompt: 'a modern tech CEO version of this person, professional headshot, studio lighting',
      negative_prompt: 'blurry, low quality, distorted',
      init_image: initImage,
      width: 1024,
      height: 1024,
      strength: STRENGTH,
      cfg_scale: 1.0,
      steps: STEPS,
      guidance: GUIDANCE,
      seed: SEED
    })

    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) {
          const totalMs = Date.now() - tGenStart
          console.log(`\n✓ Image generated in ${(totalMs / 1000).toFixed(1)}s (includes VAE encode/decode)`)
          fs.writeFileSync(outputImagePath, data)
          console.log(`✓ Saved to: ${outputImagePath}`)
          gpuMemory().then((gpusPost) => {
            if (gpusPost.length) console.log('  GPU post : ' + fmtGpus(gpusPost))
          })
          console.log('\nFor comparison, run the Q8 version:')
          console.log('  bare examples/img2img-flux2.js')
        } else if (typeof data === 'string') {
          try {
            const tick = JSON.parse(data)
            if ('step' in tick && 'total' in tick) {
              const now = Date.now()
              const stepMs = now - lastStepTime
              lastStepTime = now
              const wallMs = now - tGenStart
              const bar = makeBar(tick.step, tick.total, 30)
              console.log(`  ${bar} ${String(tick.step).padStart(2)}/${tick.total} | ${(stepMs / 1000).toFixed(2)}s/step | wall ${(wallMs / 1000).toFixed(1)}s`)
            }
          } catch (_) {}
        }
      })
      .await()

    console.log('\nDone!')
  } catch (error) {
    console.error('Error:', error)
  } finally {
    await model.unload()
  }
}

main()
