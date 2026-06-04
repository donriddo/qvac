#!/usr/bin/env node
'use strict'

// Unified benchmark report renderer for the Qwen3.5 perf benchmark.
//
// Reads perf JSON from --dir (recursively) and renders ONE markdown report:
//   - header with addon version, prompt size, runs-per-config, GPU
//   - one table per device: Config | TTFT (ms) | TPS | ppTPS | Tokens
//   - optional Δ columns when --compare-dir is provided (cross-run regression)
//   - a closing "best config per device" summary
//
// Two input schemas are normalised:
//   desktop sweep:  { models:[{modelId, cases:[{quantization, runtimeConfig,
//                    metrics:{ttftMsMean,tpsMean,ppTpsMean,promptTokens,
//                    generatedTokens}, status, isBaseline}]}], repeats, ... }
//   mobile report:  { addon, device:{name}, results:[{test, metrics:{ttft_ms,
//                    tps, pp_tps, generated_tokens, prompt_tokens}}] }

const fs = require('fs')
const path = require('path')

function parseArgs (argv) {
  const a = {
    dir: null,
    output: null,
    desktopDevice: 'Desktop (linux-x64 GPU)',
    addonVersion: null,
    compareDir: null
  }
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i]
    if (t === '--dir') a.dir = argv[++i]
    else if (t === '--output') a.output = argv[++i]
    else if (t === '--desktop-device') a.desktopDevice = argv[++i]
    else if (t === '--addon-version') a.addonVersion = argv[++i]
    else if (t === '--compare-dir') a.compareDir = argv[++i]
  }
  if (!a.dir) {
    throw new Error(
      'usage: render-report.js --dir <path> [--output <md>] ' +
      '[--desktop-device <name>] [--addon-version <ver>] [--compare-dir <path>]'
    )
  }
  return a
}

function walkJson (dir) {
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...walkJson(p))
    else if (entry.name.endsWith('.json')) out.push(p)
  }
  return out
}

function num (v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function int (v) {
  const n = num(v)
  return n !== null ? Math.round(n) : null
}

// Collect metadata and rows from all files in a directory.
// Returns { rows, meta } where meta = { addonVersion, repeats, promptTokens }.
function loadDir (dir, desktopDevice) {
  const files = walkJson(dir)
  const meta = { addonVersion: null, repeats: null, promptTokens: null }
  let rows = []
  for (const f of files) {
    const r = rowsFromFile(f, desktopDevice, meta)
    rows.push(...r)
  }
  rows = dedupe(rows)
  return { rows, meta }
}

// Normalise any report file into rows: { device, config, ttft, tps, ppTps, tokens, crashed }
// Also fills in meta fields when found.
function rowsFromFile (file, desktopDevice, meta) {
  let doc
  try { doc = JSON.parse(fs.readFileSync(file, 'utf8')) } catch { return [] }
  const rows = []

  // Desktop sweep schema
  if (Array.isArray(doc.models) && doc.models.length && Array.isArray(doc.models[0].cases)) {
    if (num(doc.repeats) !== null && meta.repeats === null) meta.repeats = doc.repeats
    for (const model of doc.models) {
      for (const c of model.cases) {
        if (c.isBaseline) continue
        const rc = c.runtimeConfig || {}
        const config = configLabel({
          model: `${model.modelId}-${c.quantization}`,
          backend: rc.device,
          rb: rc['reasoning-budget'],
          ck: rc['cache-type-k'],
          cv: rc['cache-type-v']
        })
        const m = c.metrics || {}
        if (int(m.promptTokens) !== null && meta.promptTokens === null) {
          meta.promptTokens = int(m.promptTokens)
        }
        const crashed = c.status && c.status !== 'ok' && c.status !== 'partial-failure'
        rows.push({
          device: desktopDevice,
          config,
          ttft: num(m.ttftMsMean),
          tps: num(m.tpsMean),
          ppTps: num(m.ppTpsMean),
          tokens: int(m.generatedTokens),
          crashed: !!crashed
        })
      }
    }
    return rows
  }

  // Mobile perf-report schema
  if (doc.device && Array.isArray(doc.results)) {
    if (doc.addon && meta.addonVersion === null) meta.addonVersion = doc.addon
    const device = (doc.device.name || 'unknown').trim()
    for (const r of doc.results) {
      const m = r.metrics || {}
      if (int(m.prompt_tokens) !== null && meta.promptTokens === null) {
        meta.promptTokens = int(m.prompt_tokens)
      }
      const crashed = (r.status && String(r.status).toLowerCase() === 'crashed') ||
        (num(m.ttft_ms) === null && num(m.tps) === null && num(m.pp_tps) === null)
      rows.push({
        device,
        config: r.test || '(unknown)',
        ttft: num(m.ttft_ms),
        tps: num(m.tps),
        ppTps: num(m.pp_tps),
        tokens: int(m.generated_tokens),
        crashed: !!crashed
      })
    }
    return rows
  }

  return rows
}

function configLabel ({ model, backend, rb, ck, cv }) {
  const parts = [`[${model}]`]
  if (backend) parts.push(`[${backend}]`)
  if (rb !== undefined && rb !== null && rb !== '') parts.push(`[rb=${rb}]`)
  if (ck || cv) parts.push(ck === cv ? `[kv=${ck}]` : `[kv=${ck || '?'}/${cv || '?'}]`)
  return parts.join(' ')
}

function fmt (v, decimals = 2) {
  if (v === null) return '-'
  return (Math.round(v * Math.pow(10, decimals)) / Math.pow(10, decimals)).toFixed(decimals)
}

function fmtDelta (v) {
  if (v === null) return '-'
  const sign = v >= 0 ? '+' : ''
  return `${sign}${fmt(v)}`
}

function dedupe (rows) {
  const byKey = new Map()
  for (const r of rows) {
    const k = `${r.device}@@${r.config}`
    const prev = byKey.get(k)
    if (!prev || (prev.crashed && !r.crashed)) byKey.set(k, r)
  }
  return [...byKey.values()]
}

function buildBaselineMap (baseRows) {
  const m = new Map()
  for (const r of baseRows) m.set(`${r.device}@@${r.config}`, r)
  return m
}

function render (rows, desktopDevice, meta, addonVersionArg, baselineMap) {
  const byDevice = new Map()
  for (const r of rows) {
    if (!byDevice.has(r.device)) byDevice.set(r.device, [])
    byDevice.get(r.device).push(r)
  }
  const devices = [...byDevice.keys()].sort((a, b) => {
    if (a === desktopDevice) return -1
    if (b === desktopDevice) return 1
    return a.localeCompare(b)
  })

  const addonVersion = addonVersionArg || meta.addonVersion || null
  const comparing = baselineMap !== null

  const lines = []
  lines.push('# Qwen3.5 Benchmark Results')
  lines.push('')

  // Header metadata block
  const headerParts = []
  if (addonVersion) headerParts.push(`**Addon:** \`${addonVersion}\``)
  if (meta.promptTokens !== null) headerParts.push(`**Prompt:** ${meta.promptTokens} tokens`)
  if (meta.repeats !== null) headerParts.push(`**Runs per config:** ${meta.repeats}`)
  if (headerParts.length) {
    lines.push(headerParts.join(' · '))
    lines.push('')
  }

  lines.push(
    'Metrics are addon `runtimeStats`: ' +
    'TTFT = time to first token (ms), TPS = decode tokens/sec, ' +
    'ppTPS = prefill tokens/sec, Tokens = generated tokens.' +
    (comparing ? ' Δ columns show current minus baseline.' : '') +
    ' `Crashed` = configuration crashed or produced no output.'
  )
  lines.push('')

  const hasTokens = rows.some(r => r.tokens !== null)

  for (const device of devices) {
    const items = byDevice.get(device).slice().sort((a, b) => a.config.localeCompare(b.config))
    lines.push(`## ${device}`)
    lines.push('')

    if (comparing) {
      const hdr = hasTokens
        ? '| Config | TTFT (ms) | Δ TTFT | TPS | Δ TPS | ppTPS | Δ ppTPS | Tokens |'
        : '| Config | TTFT (ms) | Δ TTFT | TPS | Δ TPS | ppTPS | Δ ppTPS |'
      const sep = hasTokens
        ? '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'
        : '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'
      lines.push(hdr)
      lines.push(sep)
      for (const r of items) {
        const b = baselineMap.get(`${r.device}@@${r.config}`)
        if (r.crashed) {
          const crash = hasTokens
            ? `| ${r.config} | Crashed | - | Crashed | - | Crashed | - | - |`
            : `| ${r.config} | Crashed | - | Crashed | - | Crashed | - |`
          lines.push(crash)
        } else {
          const dTtft = (b && !b.crashed && r.ttft !== null && b.ttft !== null) ? r.ttft - b.ttft : null
          const dTps = (b && !b.crashed && r.tps !== null && b.tps !== null) ? r.tps - b.tps : null
          const dPp = (b && !b.crashed && r.ppTps !== null && b.ppTps !== null) ? r.ppTps - b.ppTps : null
          const row = hasTokens
            ? `| ${r.config} | ${fmt(r.ttft)} | ${fmtDelta(dTtft)} | ${fmt(r.tps)} | ${fmtDelta(dTps)} | ${fmt(r.ppTps)} | ${fmtDelta(dPp)} | ${r.tokens !== null ? r.tokens : '-'} |`
            : `| ${r.config} | ${fmt(r.ttft)} | ${fmtDelta(dTtft)} | ${fmt(r.tps)} | ${fmtDelta(dTps)} | ${fmt(r.ppTps)} | ${fmtDelta(dPp)} |`
          lines.push(row)
        }
      }
    } else {
      const hdr = hasTokens
        ? '| Config | TTFT (ms) | TPS | ppTPS | Tokens |'
        : '| Config | TTFT (ms) | TPS | ppTPS |'
      const sep = hasTokens
        ? '| --- | ---: | ---: | ---: | ---: |'
        : '| --- | ---: | ---: | ---: |'
      lines.push(hdr)
      lines.push(sep)
      for (const r of items) {
        if (r.crashed) {
          lines.push(hasTokens
            ? `| ${r.config} | Crashed | Crashed | Crashed | - |`
            : `| ${r.config} | Crashed | Crashed | Crashed |`)
        } else {
          lines.push(hasTokens
            ? `| ${r.config} | ${fmt(r.ttft)} | ${fmt(r.tps)} | ${fmt(r.ppTps)} | ${r.tokens !== null ? r.tokens : '-'} |`
            : `| ${r.config} | ${fmt(r.ttft)} | ${fmt(r.tps)} | ${fmt(r.ppTps)} |`)
        }
      }
    }
    lines.push('')
  }

  lines.push('## Best configuration per device')
  lines.push('')
  lines.push('| Device | Highest TPS | Highest ppTPS |')
  lines.push('| --- | --- | --- |')
  for (const device of devices) {
    const ok = byDevice.get(device).filter(r => !r.crashed)
    const bestTps = ok.filter(r => r.tps !== null).sort((a, b) => b.tps - a.tps)[0]
    const bestPp = ok.filter(r => r.ppTps !== null).sort((a, b) => b.ppTps - a.ppTps)[0]
    const tpsCell = bestTps ? `${bestTps.config} — ${fmt(bestTps.tps)}` : '-'
    const ppCell = bestPp ? `${bestPp.config} — ${fmt(bestPp.ppTps)}` : '-'
    lines.push(`| ${device} | ${tpsCell} | ${ppCell} |`)
  }
  lines.push('')
  return lines.join('\n') + '\n'
}

function main () {
  const args = parseArgs(process.argv)

  const { rows, meta } = loadDir(args.dir, args.desktopDevice)

  let baselineMap = null
  if (args.compareDir) {
    const { rows: baseRows } = loadDir(args.compareDir, args.desktopDevice)
    baselineMap = buildBaselineMap(baseRows)
  }

  if (rows.length === 0) {
    const msg = 'No benchmark results found.\n'
    if (args.output) fs.writeFileSync(args.output, msg)
    else process.stdout.write(msg)
    return
  }

  const md = render(rows, args.desktopDevice, meta, args.addonVersion, baselineMap)
  if (args.output) fs.writeFileSync(args.output, md)
  else process.stdout.write(md)
}

main()
