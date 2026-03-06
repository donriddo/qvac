#!/usr/bin/env bun
/**
 * Compare two API version directories and generate a migration guide MDX.
 *
 * Reads the per-function MDX files produced by generate-api-docs.ts,
 * diffs them by name/signature/parameters/return type, and writes a
 * migration guide MDX with added, removed, and changed sections.
 *
 * Usage:
 *   bun run scripts/compare-api-versions.ts <old-version> <new-version> [--output <path>]
 *
 * Examples:
 *   bun run scripts/compare-api-versions.ts 0.6.0 0.7.0
 *   bun run scripts/compare-api-versions.ts 0.6.0 0.7.0 --output ./migration-guide.mdx
 */

import * as fs from "fs/promises";
import * as path from "path";

interface ParameterInfo {
  name: string;
  type: string;
  required: boolean;
}

interface FunctionInfo {
  name: string;
  signature: string;
  description: string;
  parameters: ParameterInfo[];
  returnType: string;
}

interface ChangedFunction {
  name: string;
  old: FunctionInfo;
  new: FunctionInfo;
  changes: string[];
}

interface VersionDiff {
  added: FunctionInfo[];
  removed: FunctionInfo[];
  changed: ChangedFunction[];
  unchanged: string[];
}

const API_BASE = path.join(process.cwd(), "content", "docs", "sdk", "api");

function parseFunctionFromMDX(content: string, fileName: string): FunctionInfo {
  const name = fileName.replace(/\.mdx$/, "");

  const sigMatch = content.match(/```typescript\n(function [^\n]+)\n```/);
  const signature = sigMatch?.[1] ?? "";

  const descMatch = content.match(/## Description\n\n([\s\S]*?)(?=\n## )/);
  const description = descMatch?.[1]?.trim() ?? "";

  const parameters: ParameterInfo[] = [];
  const paramSection = content.match(
    /## Parameters\n\n\|[^\n]+\n\|[^\n]+\n([\s\S]*?)(?=\n## )/
  );
  if (paramSection) {
    for (const row of paramSection[1].trim().split("\n")) {
      const cells = row
        .split("|")
        .map((c) => c.trim())
        .filter(Boolean);
      if (cells.length >= 3) {
        parameters.push({
          name: cells[0].replace(/`/g, ""),
          type: cells[1].replace(/`/g, ""),
          required: cells[2] === "✓",
        });
      }
    }
  }

  const retMatch = content.match(/## Returns\n\n```typescript\n([\s\S]*?)\n```/);
  const returnType = retMatch?.[1]?.trim() ?? "unknown";

  return { name, signature, description, parameters, returnType };
}

async function loadVersionFunctions(
  version: string
): Promise<Map<string, FunctionInfo>> {
  const versionDir = path.join(API_BASE, `v${version}`);

  try {
    await fs.stat(versionDir);
  } catch {
    throw new Error(
      `Version directory not found: ${versionDir}\n` +
        `Run docs:generate-api for this version first, or check ${API_BASE} for available versions.`
    );
  }

  const entries = await fs.readdir(versionDir);
  const mdxFiles = entries.filter(
    (f) => f.endsWith(".mdx") && f !== "index.mdx"
  );

  const functions = new Map<string, FunctionInfo>();
  await Promise.all(
    mdxFiles.map(async (file) => {
      const content = await fs.readFile(path.join(versionDir, file), "utf-8");
      const fn = parseFunctionFromMDX(content, file);
      functions.set(fn.name, fn);
    })
  );

  return functions;
}

function diffVersions(
  oldFns: Map<string, FunctionInfo>,
  newFns: Map<string, FunctionInfo>
): VersionDiff {
  const added: FunctionInfo[] = [];
  const removed: FunctionInfo[] = [];
  const changed: ChangedFunction[] = [];
  const unchanged: string[] = [];

  for (const [name, newFn] of newFns) {
    const oldFn = oldFns.get(name);
    if (!oldFn) {
      added.push(newFn);
      continue;
    }

    const changes: string[] = [];

    if (oldFn.signature !== newFn.signature) {
      changes.push("signature");
    }

    if (oldFn.parameters.length !== newFn.parameters.length) {
      changes.push("parameter count");
    } else {
      const paramsDiffer = oldFn.parameters.some((op, i) => {
        const np = newFn.parameters[i];
        return op.name !== np.name || op.type !== np.type || op.required !== np.required;
      });
      if (paramsDiffer) changes.push("parameters");
    }

    if (oldFn.returnType !== newFn.returnType) {
      changes.push("return type");
    }

    if (changes.length > 0) {
      changed.push({ name, old: oldFn, new: newFn, changes });
    } else {
      unchanged.push(name);
    }
  }

  for (const [name, oldFn] of oldFns) {
    if (!newFns.has(name)) {
      removed.push(oldFn);
    }
  }

  const byName = (a: { name: string }, b: { name: string }) =>
    a.name.localeCompare(b.name);
  added.sort(byName);
  removed.sort(byName);
  changed.sort(byName);
  unchanged.sort();

  return { added, removed, changed, unchanged };
}

function renderParamTable(params: ParameterInfo[]): string {
  if (params.length === 0) return "_None_\n";
  let table = "| Name | Type | Required? |\n| --- | --- | :---: |\n";
  for (const p of params) {
    table += `| \`${p.name}\` | \`${p.type}\` | ${p.required ? "✓" : "✗"} |\n`;
  }
  return table;
}

function generateMigrationMDX(
  oldVersion: string,
  newVersion: string,
  diff: VersionDiff
): string {
  const lines: string[] = [];
  const totalChanges = diff.added.length + diff.removed.length + diff.changed.length;

  lines.push(`---
title: "Migration: v${oldVersion} → v${newVersion}"
description: "API changes and migration steps from QVAC SDK v${oldVersion} to v${newVersion}"
---

## Summary

| Change Type | Count |
| --- | :---: |
| Added functions | ${diff.added.length} |
| Removed functions | ${diff.removed.length} |
| Changed functions | ${diff.changed.length} |
| Unchanged functions | ${diff.unchanged.length} |
| **Total API surface** | **${diff.unchanged.length + diff.changed.length + diff.added.length}** |`);

  if (diff.added.length > 0) {
    lines.push(`
## Added Functions

The following ${diff.added.length} function${diff.added.length === 1 ? " was" : "s were"} added in v${newVersion}.
`);

    for (const fn of diff.added) {
      lines.push(`### \`${fn.name}()\`\n`);
      if (fn.description && fn.description !== "No description available") {
        lines.push(`${fn.description}\n`);
      }
      if (fn.signature) {
        lines.push(`\`\`\`typescript\n${fn.signature}\n\`\`\`\n`);
      }
    }
  }

  if (diff.removed.length > 0) {
    lines.push(`
## Removed Functions

The following ${diff.removed.length} function${diff.removed.length === 1 ? " was" : "s were"} removed in v${newVersion}. Update your code to remove any usage.
`);

    for (const fn of diff.removed) {
      lines.push(`### \`${fn.name}()\`\n`);
      if (fn.description && fn.description !== "No description available") {
        lines.push(`${fn.description}\n`);
      }
      if (fn.signature) {
        lines.push(`Previous signature:\n\n\`\`\`typescript\n${fn.signature}\n\`\`\`\n`);
      }
    }
  }

  if (diff.changed.length > 0) {
    lines.push(`
## Changed Functions

The following ${diff.changed.length} function${diff.changed.length === 1 ? " has" : "s have"} signature or parameter changes.
`);

    for (const entry of diff.changed) {
      lines.push(`### \`${entry.name}()\`\n`);
      lines.push(`Changes: ${entry.changes.join(", ")}\n`);

      if (entry.changes.includes("signature")) {
        lines.push(`**Before (v${oldVersion}):**\n`);
        lines.push(`\`\`\`typescript\n${entry.old.signature}\n\`\`\`\n`);
        lines.push(`**After (v${newVersion}):**\n`);
        lines.push(`\`\`\`typescript\n${entry.new.signature}\n\`\`\`\n`);
      }

      if (
        entry.changes.includes("parameters") ||
        entry.changes.includes("parameter count")
      ) {
        lines.push(`**Parameters (v${oldVersion}):**\n`);
        lines.push(renderParamTable(entry.old.parameters));
        lines.push(`**Parameters (v${newVersion}):**\n`);
        lines.push(renderParamTable(entry.new.parameters));
      }

      if (entry.changes.includes("return type")) {
        lines.push(
          `**Return type:** \`${entry.old.returnType}\` → \`${entry.new.returnType}\`\n`
        );
      }
    }
  }

  if (totalChanges === 0) {
    lines.push(`
## No Changes

No API function changes detected between v${oldVersion} and v${newVersion}.
`);
  }

  return lines.join("\n").trimEnd() + "\n";
}

async function compareApiVersions(
  oldVersion: string,
  newVersion: string,
  outputPath?: string
) {
  for (const v of [oldVersion, newVersion]) {
    if (!/^\d+\.\d+\.\d+$/.test(v)) {
      throw new Error(
        `Invalid version format: "${v}"\nExpected semver: X.Y.Z (e.g., 0.6.1)`
      );
    }
  }

  console.log(`🔍 Comparing API versions: v${oldVersion} → v${newVersion}`);

  const oldFns = await loadVersionFunctions(oldVersion);
  console.log(`✓ Loaded v${oldVersion}: ${oldFns.size} functions`);

  const newFns = await loadVersionFunctions(newVersion);
  console.log(`✓ Loaded v${newVersion}: ${newFns.size} functions`);

  const diff = diffVersions(oldFns, newFns);
  console.log(
    `✓ Diff: +${diff.added.length} added, -${diff.removed.length} removed, ~${diff.changed.length} changed`
  );

  const mdx = generateMigrationMDX(oldVersion, newVersion, diff);

  const outFile =
    outputPath ??
    path.join(API_BASE, `migration-v${oldVersion}-to-v${newVersion}.mdx`);
  const outDir = path.dirname(path.resolve(outFile));
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(outFile, mdx, "utf-8");

  console.log(`✅ Migration guide generated: ${outFile}`);
  console.log(`   Added:     ${diff.added.length} functions`);
  console.log(`   Removed:   ${diff.removed.length} functions`);
  console.log(`   Changed:   ${diff.changed.length} functions`);
  console.log(`   Unchanged: ${diff.unchanged.length} functions`);
}

// CLI
const args = process.argv.slice(2);

if (args.includes("--help") || args.includes("-h")) {
  console.log(
    "Usage: bun run scripts/compare-api-versions.ts <old-version> <new-version> [flags]\n"
  );
  console.log("Compares two versioned API directories and generates a migration guide MDX.\n");
  console.log("Flags:");
  console.log("  --output <path>  Custom output path for the migration guide MDX\n");
  console.log("Examples:");
  console.log("  bun run scripts/compare-api-versions.ts 0.6.0 0.7.0");
  console.log(
    "  bun run scripts/compare-api-versions.ts 0.6.0 0.7.0 --output ./migration.mdx"
  );
  process.exit(0);
}

const outputIdx = args.indexOf("--output");
const outputPath = outputIdx !== -1 ? args[outputIdx + 1] : undefined;
const versions = args.filter(
  (arg, i) => !arg.startsWith("--") && args[i - 1] !== "--output"
);

if (versions.length !== 2) {
  console.error("❌ Error: Two version arguments required\n");
  console.error(
    "Usage: bun run scripts/compare-api-versions.ts <old-version> <new-version> [--output <path>]\n"
  );
  console.error("Examples:");
  console.error("  bun run scripts/compare-api-versions.ts 0.6.0 0.7.0");
  console.error(
    "  bun run scripts/compare-api-versions.ts 0.6.0 0.7.0 --output ./migration.mdx"
  );
  process.exit(1);
}

compareApiVersions(versions[0], versions[1], outputPath).catch((error) => {
  console.error("❌ Error comparing API versions:", error.message);
  if (error.stack) console.error("\nStack trace:", error.stack);
  process.exit(1);
});
