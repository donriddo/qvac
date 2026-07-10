import {
  createRequestRegistry as createRegistry,
  type RequestRegistry
} from '@/server/bare/runtime/request-registry'

/**
 * Worker-process singleton. Every long-running request in this Bare
 * worker registers under this registry, so a `cancel({ requestId })` RPC
 * can find its target without the caller needing to know which plugin /
 * handler owns the request.
 *
 * Exposed alongside `createRequestRegistry()` rather than replacing it so
 * unit tests can spin up isolated registries without contaminating the
 * shared instance. On first use the singleton registers the SDK's
 * baseline concurrency policies.
 */
let registry: RequestRegistry | null = null

// Completions that persist a disk KV-cache share process-global cache
// bookkeeping (the `.bin` file plus the `initializedCaches` /
// `cachedMessageCounts` maps) and would corrupt it if two decoded at once.
// The completionStream handler routes them onto this dedicated cap-1 lane so
// they serialize against each other, while plain completions of the same kind
// go N-way on the default `completion` lane.
export const LLAMACPP_COMPLETION_CACHED_SLOT_GROUP = 'llamacppCompletionCached'

function installDefaultPolicies(r: RequestRegistry): void {
  // Single completions on one model decode concurrently (continuous batching)
  // up to the model's own `parallel` slot count. That count is per-model, so
  // the completionStream handler passes it per request as
  // `maxConcurrentPerModel`; the value here is only the fallback for a caller
  // that supplies none (serialize, matching a model loaded without `parallel`).
  // Overflow waits FIFO; the depth cap bounds queue memory.
  r.policy({
    kind: 'completion',
    maxConcurrentPerModel: 1,
    onOverflow: 'queue',
    maxQueueDepthPerModel: 64
  })
  // A batch already occupies the model's sequence slots, so it stays
  // one-at-a-time per model on its own lane. It no longer shares a lane with
  // `completion`: a batch and single completions may overlap, and the addon's
  // multi-job scheduler admits them together, queuing any surplus sequences.
  r.policy({
    kind: 'batchCompletion',
    maxConcurrentPerModel: 1,
    onOverflow: 'queue',
    maxQueueDepthPerModel: 64
  })
}

export function getRequestRegistry(): RequestRegistry {
  if (!registry) {
    registry = createRegistry()
    installDefaultPolicies(registry)
  }
  return registry
}

export { createRegistry as createRequestRegistry }
