const SHARD_PATTERN = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/;

/**
 * Expand a GGUF model path into its constituent shard paths.
 *
 * Sharded GGUF models follow the convention `<basename>-NNNNN-of-MMMMM.gguf`
 * with a sibling `<basename>.tensors.txt` metadata file. This helper mirrors
 * the C++ `GGUFShards::expandGGUFIntoShards` logic so the SDK can pass the
 * full ordered list of files to the addon's `files.model` argument, which is
 * the contract introduced by the addon-loader-abstraction refactor.
 *
 * Order matters: the tensors.txt file is yielded first, followed by each
 * shard from `00001-of-NNNNN` through `NNNNN-of-NNNNN`. The addon streams
 * weights in this order and picks the first shard-matching entry (the
 * `-00001-of-NNNNN.gguf` file) as the resolved model path for native
 * loading; the `.tensors.txt` companion is consumed by the weight-streaming
 * layer but is not used as the primary path.
 *
 * Non-sharded models (or paths whose filename does not match the pattern)
 * are returned as a single-element array containing the input path
 * unchanged.
 *
 * Pure string manipulation — no filesystem or runtime-specific path module.
 * Handles both POSIX (`/`) and Windows (`\`) separators based on whichever
 * appears in the input.
 */
export function expandGGUFIntoShards(modelPath: string): string[] {
  const lastSep = Math.max(
    modelPath.lastIndexOf("/"),
    modelPath.lastIndexOf("\\"),
  );
  const dir = lastSep >= 0 ? modelPath.slice(0, lastSep) : "";
  const sep = lastSep >= 0 ? modelPath.charAt(lastSep) : "/";
  const filename = lastSep >= 0 ? modelPath.slice(lastSep + 1) : modelPath;

  const match = filename.match(SHARD_PATTERN);
  if (!match || !match[1] || !match[3]) return [modelPath];

  const basename = match[1];
  const totalShards = Number.parseInt(match[3], 10);
  if (!Number.isFinite(totalShards) || totalShards <= 0) return [modelPath];

  const join = (name: string) => (dir ? `${dir}${sep}${name}` : name);
  const shards: string[] = [join(`${basename}.tensors.txt`)];
  const totalDigits = String(totalShards).padStart(5, "0");
  for (let i = 1; i <= totalShards; i++) {
    shards.push(
      join(`${basename}-${String(i).padStart(5, "0")}-of-${totalDigits}.gguf`),
    );
  }
  return shards;
}
