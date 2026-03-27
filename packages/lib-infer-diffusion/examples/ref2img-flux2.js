'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const FilesystemDL = require('@qvac/dl-filesystem')
const ImgStableDiffusion = require('../index')

/**
 * FLUX2-klein ref2img example — In-context conditioning (matches Iris C engine)
 *
 * Unlike img2img (which noises the encoded image and denoises it), ref2img
 * passes the reference image as separate tokens the FLUX transformer attends
 * to via joint attention. The target starts from pure noise, so the model can
 * reason about the reference's features (skin tone, structure, etc.) while
 * generating a fully new image — eliminating the racial bias seen with
 * traditional img2img.
 */

async function main () {
  const modelDir = path.join(__dirname, '../models')
  const inputImagePath = path.join(__dirname, '../temp/nik_headshot_832.jpeg')
  const outputImagePath = path.join(__dirname, '../temp/nik_headshot_832_ref2img.png')

  if (!fs.existsSync(inputImagePath)) {
    console.error(`Error: Input image not found at ${inputImagePath}`)
    process.exit(1)
  }

  console.log('Loading FLUX2-klein model...')

  const loader = new FilesystemDL({ dirPath: modelDir })

  const model = new ImgStableDiffusion(
    {
      loader,
      logger: console,
      diskPath: modelDir,
      modelName: 'flux-2-klein-4b-Q8_0.gguf',
      llmModel: 'Qwen3-4B-Q4_K_M.gguf',
      vaeModel: 'flux2-vae.safetensors'
    },
    {
      threads: 4,
      device: 'gpu',
      prediction: 'flux2_flow'
    }
  )

  try {
    await model.load()
    console.log('Model loaded!')

    const refImage = fs.readFileSync(inputImagePath)
    console.log(`Reference image: ${refImage.length} bytes`)

    const STEPS = 15
    const GUIDANCE = 9.0
    const SEED = -1

    console.log(`\n=== ref2img: In-Context Conditioning (Iris-equivalent) ===`)
    console.log(`  Model    : flux-2-klein-4b-Q8_0.gguf`)
    console.log(`  Mode     : ref2img (reference tokens + pure noise target)`)
    console.log(`  Steps    : ${STEPS} (ALL steps, no strength truncation)`)
    console.log(`  Guidance : ${GUIDANCE}`)
    console.log(`  Seed     : ${SEED}`)
    console.log(`  Note     : Reference image is attended to via joint attention,`)
    console.log(`             NOT mixed with noise. This preserves features.\n`)

    const tGenStart = Date.now()
    let lastStepTime = tGenStart

    const response = await model.run({
      prompt: 'a soccer player version of this photo, professional headshot, with shaved buzz cut.',
      negative_prompt: 'blurry, low quality, distorted',
      init_image: refImage,
      cfg_scale: 1.0,
      steps: STEPS,
      guidance: GUIDANCE,
      seed: SEED
    })

    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) {
          const totalMs = Date.now() - tGenStart
          console.log(`\n✓ Image generated in ${(totalMs / 1000).toFixed(1)}s`)
          fs.writeFileSync(outputImagePath, data)
          console.log(`✓ Saved to: ${outputImagePath}`)
          console.log(`\nCompare with traditional img2img:`)
          console.log(`  bare examples/img2img-flux2.js`)
          console.log(`Compare with Iris:`)
          console.log(`  bash scripts/img2img-flux-iris.sh`)
        } else if (typeof data === 'string') {
          try {
            const tick = JSON.parse(data)
            if ('step' in tick && 'total' in tick) {
              const now = Date.now()
              const stepMs = now - lastStepTime
              lastStepTime = now
              const wallMs = now - tGenStart
              process.stdout.write(
                `\r  step ${tick.step}/${tick.total} | step took ${(stepMs / 1000).toFixed(1)}s | wall ${(wallMs / 1000).toFixed(1)}s elapsed  `
              )
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
    await loader.close()
  }
}

main()
