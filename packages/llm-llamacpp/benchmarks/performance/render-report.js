#!/usr/bin/env node
'use strict'

// Unified benchmark report renderer for the Qwen3.5 perf benchmark.
//
// Reads perf JSON from --dir (recursively) and renders ONE markdown report
// with the same shape for desktop and mobile, per Gianfranco's request:
//   - one table per device, columns: Config | TTFT (ms) | TPS | ppTPS
//     (only the addon runtimeStats; no Total/Prefill/Decode/Platform/Mean)
//   - a closing "best config per device" summary (highest TPS, highest ppTPS)
//   - crashed combos render as `Crashed` instead of metrics
//
// Two input schemas are normalized:
//   - desktop sweep:  { models: [ { modelId, cases: [ { quantization,
//                       runtimeConfig, metrics:{ttftMsMean,tpsMean,ppTpsMean},
//                       status, isBaseline } ] } ]   (one logical device: the
//                       desktop GPU runner)
//   - mobile report:  { device:{name}, results:[ { test, status,
//                       metrics:{ttft_ms,tps,pp_tps} } ] }

const fs = require('fs')
const path = require('path')

function parseArgs (argv) {
  const a = { dir: null, output: null, desktopDevice: 'Desktop (linux-x64 GPU)' }
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i]
    if (t === '--dir') a.dir = argv[++i]
    else if (t === '--output') a.output = argv[++i]
    else if (t === '--desktop-device') a.desktopDevice = argv[++i]
  }
  if (!a.dir) throw new Error('usage: render-report.js --dir <path> [--output <md>] [--desktop-device <name>]')
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

// Normalize any report file into rows: { device, config, ttft, tps, ppTps, crashed }
function rowsFromFile (file, desktopDevice) {
  let doc
  try { doc = JSON.parse(fs.readFileSync(file, 'utf8')) } catch { return [] }
  const rows = []

  // Desktop sweep schema
  if (Array.isArray(doc.models) && doc.models.length && Array.isArray(doc.models[0].cases)) {
    for (const model of doc.models) {
      for (const c of model.cases) {
        if (c.isBaseline) continue // baseline duplicates a swept combo
        const rc = c.runtimeConfig || {}
        const config = configLabel({
          model: `${model.modelId}-${c.quantization}`,
          backend: rc.device,
          rb: rc['reasoning-budget'],
          ck: rc['cache-type-k'],
          cv: rc['cache-type-v']
        })
        const m = c.metrics || {}
        const crashed = c.status && c.status !== 'ok' && c.status !== 'partial-failure'
        rows.push({
          device: desktopDevice,
          config,
          ttft: num(m.ttftMsMean),
          tps: num(m.tpsMean),
          ppTps: num(m.ppTpsMean),
          crashed: !!crashed
        })
      }
    }
    return rows
  }

  // Mobile perf-report schema
  if (doc.device && Array.isArray(doc.results)) {
    const device = (doc.device.name || 'unknown').trim()
    for (const r of doc.results) {
      const m = r.metrics || {}
      const crashed = (r.status && String(r.status).toLowerCase() === 'crashed') ||
        (num(m.ttft_ms) === null && num(m.tps) === null && num(m.pp_tps) === null)
      rows.push({
        device,
        config: r.test || '(unknown)',
        ttft: num(m.ttft_ms),
        tps: num(m.tps),
        ppTps: num(m.pp_tps),
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
  if (ck || cv) parts.push(`[kv=${ck || '?'}/${cv || '?'}]`)
  return parts.join(' ')
}

function fmt (v) {
  return v === null ? '-' : (Math.round(v * 100) / 100)
}

function dedupe (rows) {
  // last write wins per (device, config); prefer a non-crashed row with metrics
  const byKey = new Map()
  for (const r of rows) {
    const k = `${r.device}@@${r.config}`
    const prev = byKey.get(k)
    if (!prev || (prev.crashed && !r.crashed)) byKey.set(k, r)
  }
  return [...byKey.values()]
}

function render (rows, desktopDevice) {
  const byDevice = new Map()
  for (const r of rows) {
    if (!byDevice.has(r.device)) byDevice.set(r.device, [])
    byDevice.get(r.device).push(r)
  }
  // desktop device first, then the rest alphabetically
  const devices = [...byDevice.keys()].sort((a, b) => {
    if (a === desktopDevice) return -1
    if (b === desktopDevice) return 1
    return a.localeCompare(b)
  })

  const lines = []
  lines.push('# Qwen3.5 Benchmark Results')
  lines.push('')
  lines.push('Metrics are the addon `runtimeStats`: TTFT (time to first token, ms), TPS (decode tokens/sec), ppTPS (prefill tokens/sec). `Crashed` means the configuration crashed or produced no output on that device.')
  lines.push('')

  for (const device of devices) {
    const items = byDevice.get(device).slice().sort((a, b) => a.config.localeCompare(b.config))
    lines.push(`## ${device}`)
    lines.push('')
    lines.push('| Config | TTFT (ms) | TPS | ppTPS |')
    lines.push('| --- | ---: | ---: | ---: |')
    for (const r of items) {
      if (r.crashed) lines.push(`| ${r.config} | Crashed | Crashed | Crashed |`)
      else lines.push(`| ${r.config} | ${fmt(r.ttft)} | ${fmt(r.tps)} | ${fmt(r.ppTps)} |`)
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
  const files = walkJson(args.dir)
  let rows = []
  for (const f of files) rows.push(...rowsFromFile(f, args.desktopDevice))
  rows = dedupe(rows)
  if (rows.length === 0) {
    const msg = 'No benchmark results found.\n'
    if (args.output) fs.writeFileSync(args.output, msg)
    else process.stdout.write(msg)
    return
  }
  const md = render(rows, args.desktopDevice)
  if (args.output) fs.writeFileSync(args.output, md)
  else process.stdout.write(md)
}

main()
