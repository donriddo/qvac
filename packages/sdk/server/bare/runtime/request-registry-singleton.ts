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

// `completion` and `batchCompletion` share one admission lane per model: all
// requests — single completions and batches alike — compete for the model's
// `parallel` sequence slots first-come-first-serve, so they contend for one
// slot pool rather than each getting an independent lane.
const LLAMACPP_COMPLETION_SLOT_GROUP = 'llamacppCompletion'

// Completions that persist a disk KV-cache share process-global cache
// bookkeeping (the `.bin` file plus the `initializedCaches` /
// `cachedMessageCounts` maps) and would corrupt it if two decoded at once.
// When the model runs N-way (`parallel > 1`) the handler routes them onto this
// dedicated cap-1 lane so they serialize against each other while plain
// completions go concurrent; at `parallel = 1` everything is already serial on
// the shared lane, so this lane is unused.
export const LLAMACPP_COMPLETION_CACHED_SLOT_GROUP = 'llamacppCompletionCached'

function installDefaultPolicies(r: RequestRegistry): void {
  // completion + batchCompletion admit up to the model's own `parallel` jobs
  // concurrently (continuous batching), then queue the surplus FCFS. The cap is
  // per-model, so the handlers pass it per request as `maxConcurrentPerModel`;
  // the value here is only the fallback for a caller that supplies none. The
  // shared slot group makes a completion and a batch on the same model contend
  // for one pool; the depth cap bounds queue memory.
  r.policy({
    kind: 'completion',
    maxConcurrentPerModel: 1,
    onOverflow: 'queue',
    maxQueueDepthPerModel: 64,
    sharedSlotGroup: LLAMACPP_COMPLETION_SLOT_GROUP
  })
  r.policy({
    kind: 'batchCompletion',
    maxConcurrentPerModel: 1,
    onOverflow: 'queue',
    maxQueueDepthPerModel: 64,
    sharedSlotGroup: LLAMACPP_COMPLETION_SLOT_GROUP
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
