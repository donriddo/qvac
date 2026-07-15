import { completion } from '@qvac/sdk'
import { ValidationHelpers, type TestResult, type Expectation } from '@tetherto/qvac-test-suite'
import { AbstractModelExecutor } from './abstract-model-executor.js'
import { completionTests } from '../../completion-tests.js'

type ResponseFormat =
  | { type: 'text' }
  | { type: 'json_object' }
  | {
      type: 'json_schema'
      json_schema: {
        name: string
        schema: Record<string, unknown>
        description?: string
        strict?: boolean
      }
    }

interface GenerationParams {
  temp?: number
  top_p?: number
  top_k?: number
  predict?: number
  seed?: number
  frequency_penalty?: number
  presence_penalty?: number
  repeat_penalty?: number
  reasoning_budget?: number
  remove_thinking_from_context?: boolean
}

interface CompletionTestParams {
  history: ReadonlyArray<{ role: string; content: string }>
  stream?: boolean
  responseFormat?: ResponseFormat
  tools?: ReadonlyArray<Record<string, unknown>>
  stopSequences?: ReadonlyArray<string>
  generationParams?: GenerationParams
}

type CompletionFnParams = Parameters<typeof completion>[0]

export class CompletionExecutor extends AbstractModelExecutor<typeof completionTests> {
  pattern = /^completion-/

  protected handlers = Object.fromEntries(
    completionTests.map((test) => {
      if (
        test.testId === 'completion-response-format-json-object' ||
        test.testId === 'completion-response-format-json-object-streaming'
      ) {
        return [test.testId, this.responseFormatJsonObject.bind(this)]
      }
      if (test.testId === 'completion-response-format-json-schema') {
        return [test.testId, this.responseFormatJsonSchema.bind(this)]
      }
      if (test.testId === 'completion-response-format-with-tools-rejected') {
        return [test.testId, this.responseFormatWithToolsRejected.bind(this)]
      }
      if (test.testId === 'completion-stats') {
        return [test.testId, this.statsVerification.bind(this)]
      }
      if (test.testId === 'completion-concurrent-requests') {
        return [test.testId, this.concurrentRequests.bind(this)]
      }
      if (test.testId === 'completion-seed-reproducibility') {
        return [test.testId, this.seedReproducibility.bind(this)]
      }
      if (test.testId === 'completion-stop-reason-length') {
        return [test.testId, this.stopReasonLength.bind(this)]
      }
      return [test.testId, this.generic.bind(this)]
    })
  ) as never

  private async runCompletion(params: CompletionTestParams): Promise<string> {
    const llmModelId = await this.resources.ensureLoaded('llm')
    const result = completion({
      modelId: llmModelId,
      ...params,
      stream: params.stream ?? false
    } as CompletionFnParams)

    if (params.stream) {
      let fullText = ''
      for await (const token of result.tokenStream) {
        fullText += token
      }
      return fullText
    }
    return result.text
  }

  async generic(params: CompletionTestParams, expectation: Expectation): Promise<TestResult> {
    const text = await this.runCompletion(params)
    return ValidationHelpers.validate(text, expectation)
  }

  // Issues several completions against the same single-slot (parallel=1) model
  // in the same tick and asserts the single-slot admission contract: the model
  // has one native slot, so the first is admitted and the rest reject with
  // RequestRejectedByPolicyError. (Load a parallel>1 model to decode them
  // concurrently instead of rejecting.)
  async concurrentRequests(
    params: CompletionTestParams,
    expectation: Expectation
  ): Promise<TestResult> {
    const llmModelId = await this.resources.ensureLoaded('llm')
    const CONCURRENCY = 3

    const settled = await Promise.allSettled(
      Array.from(
        { length: CONCURRENCY },
        () =>
          completion({
            modelId: llmModelId,
            ...params,
            stream: false
          } as CompletionFnParams).text
      )
    )

    const fulfilled = settled.filter(
      (s): s is PromiseFulfilledResult<string> => s.status === 'fulfilled'
    )
    const rejected = settled.filter((s): s is PromiseRejectedResult => s.status === 'rejected')

    if (fulfilled.length !== 1 || rejected.length !== CONCURRENCY - 1) {
      return {
        passed: false,
        output:
          `Expected single-slot shape: 1 fulfilled and ${CONCURRENCY - 1} rejected; ` +
          `got ${fulfilled.length} fulfilled and ${rejected.length} rejected`
      }
    }

    // The over-capacity requests must fail with the admission policy error,
    // not some inference error. Match the policy-rejection message (the error
    // name is not preserved across the RPC boundary).
    const wrongReject = rejected.find(
      (r) =>
        !/rejected by registry concurrency policy/i.test(
          String((r.reason as { message?: string } | undefined)?.message ?? r.reason)
        )
    )
    if (wrongReject) {
      return {
        passed: false,
        output: `A rejection was not a registry concurrency-policy rejection: ${String((wrongReject.reason as Error)?.message ?? wrongReject.reason)}`
      }
    }

    // The admitted completion must still produce a valid response.
    const validation = ValidationHelpers.validate(fulfilled[0]!.value, expectation)
    if (!validation.passed) {
      return {
        passed: false,
        output: `Admitted completion failed expectation: ${validation.output}`
      }
    }

    return {
      passed: true,
      output:
        `Single-slot admission enforced: ${fulfilled.length} completed, ` +
        `${rejected.length} rejected (of ${CONCURRENCY} issued)`
    }
  }

  // Runs the same prompt twice with a fixed seed and asserts byte-identical
  // output, proving seeded sampling is reproducible.
  async seedReproducibility(params: CompletionTestParams): Promise<TestResult> {
    const first = await this.runCompletion(params)
    const second = await this.runCompletion(params)

    if (first.length === 0) {
      return { passed: false, output: 'First run returned an empty response' }
    }
    if (first !== second) {
      return {
        passed: false,
        output:
          `Same seed produced different output.\nRun 1: ${JSON.stringify(first.slice(0, 200))}\n` +
          `Run 2: ${JSON.stringify(second.slice(0, 200))}`
      }
    }
    return {
      passed: true,
      output: `Seeded output reproducible (${first.length} chars identical across 2 runs)`
    }
  }

  async statsVerification(
    params: CompletionTestParams,
    expectation: Expectation
  ): Promise<TestResult> {
    try {
      const llmModelId = await this.resources.ensureLoaded('llm')
      const result = completion({
        modelId: llmModelId,
        ...params,
        stream: params.stream ?? false
      } as CompletionFnParams)

      const text = await result.text
      const textValidation = ValidationHelpers.validate(text, expectation)
      if (!textValidation.passed) return textValidation

      const stats = (await result.stats) as Record<string, unknown> | undefined
      if (!stats) {
        return {
          passed: false,
          output: `Completion OK but stats were undefined. Text: "${text.slice(0, 120)}"`
        }
      }
      const ttft = stats.timeToFirstToken
      const tps = stats.tokensPerSecond
      if (typeof ttft !== 'number' || typeof tps !== 'number') {
        return {
          passed: false,
          output: `Completion stats missing numeric timing fields. Got: ${JSON.stringify(stats)}`
        }
      }
      return {
        passed: true,
        output: `completion stats OK — timeToFirstToken=${ttft}, tokensPerSecond=${tps}`
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      return { passed: false, output: `completion stats failed: ${errorMsg}` }
    }
  }

  async stopReasonLength(params: CompletionTestParams): Promise<TestResult> {
    const llmModelId = await this.resources.ensureLoaded('llm')
    const run = completion({
      modelId: llmModelId,
      ...params,
      stream: false
    } as CompletionFnParams)

    const final = await run.final
    if (final.stopReason !== 'length') {
      return {
        passed: false,
        output: `Expected stopReason "length", got ${JSON.stringify(final.stopReason)}`
      }
    }
    return {
      passed: true,
      output: `stopReason is "length" as expected`
    }
  }

  async responseFormatJsonObject(params: CompletionTestParams): Promise<TestResult> {
    try {
      const text = await this.runCompletion(params)
      return validateJsonObject(text)
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      return {
        passed: false,
        output: `responseFormat json_object failed: ${errorMsg}`
      }
    }
  }

  async responseFormatJsonSchema(params: CompletionTestParams): Promise<TestResult> {
    try {
      const text = await this.runCompletion(params)
      return validatePersonSchema(text)
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      return {
        passed: false,
        output: `responseFormat json_schema failed: ${errorMsg}`
      }
    }
  }

  async responseFormatWithToolsRejected(
    params: CompletionTestParams,
    expectation: Expectation
  ): Promise<TestResult> {
    try {
      const run = completion({
        modelId: 'schema-refinement-placeholder',
        ...params,
        stream: params.stream ?? false
      } as CompletionFnParams)
      await run.text
      return {
        passed: false,
        output: 'Expected zod refinement to reject responseFormat + tools combination'
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      return ValidationHelpers.validate(errorMsg, expectation)
    }
  }
}

type JsonObject = Record<string, unknown>
type ParseObjectResult = { ok: true; obj: JsonObject } | { ok: false; failure: TestResult }

function parseJsonObject(text: string, label: string): ParseObjectResult {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error)
    return {
      ok: false,
      failure: {
        passed: false,
        output: `${label} output is not valid JSON: ${errorMsg}. Output: ${text.slice(0, 200)}`
      }
    }
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return {
      ok: false,
      failure: {
        passed: false,
        output: `${label}: expected a JSON object, got ${
          Array.isArray(parsed) ? 'array' : typeof parsed
        }: ${text.slice(0, 200)}`
      }
    }
  }
  return { ok: true, obj: parsed as JsonObject }
}

function validateJsonObject(text: string): TestResult {
  const parsed = parseJsonObject(text, 'json_object')
  if (!parsed.ok) return parsed.failure
  return {
    passed: true,
    output: `json_object OK — keys: ${Object.keys(parsed.obj).join(',') || '(none)'}`
  }
}

const PERSON_REQUIRED_KEYS: ReadonlyArray<'name' | 'age' | 'occupation'> = [
  'name',
  'age',
  'occupation'
]

function validatePersonSchema(text: string): TestResult {
  const parsed = parseJsonObject(text, 'json_schema')
  if (!parsed.ok) return parsed.failure
  const obj = parsed.obj

  if (typeof obj.name !== 'string' || obj.name.length === 0) {
    return {
      passed: false,
      output: `name must be non-empty string, got: ${JSON.stringify(obj.name)}`
    }
  }
  if (typeof obj.age !== 'number' || !Number.isInteger(obj.age)) {
    return {
      passed: false,
      output: `age must be integer, got: ${JSON.stringify(obj.age)}`
    }
  }
  if (typeof obj.occupation !== 'string' || obj.occupation.length === 0) {
    return {
      passed: false,
      output: `occupation must be non-empty string, got: ${JSON.stringify(obj.occupation)}`
    }
  }

  const actualKeys = Object.keys(obj).sort()
  const expectedKeys = [...PERSON_REQUIRED_KEYS].sort()
  const sameKeys =
    actualKeys.length === expectedKeys.length && actualKeys.every((k, i) => k === expectedKeys[i])
  if (!sameKeys) {
    return {
      passed: false,
      output:
        `additionalProperties:false violated. Expected exactly [${expectedKeys.join(',')}], ` +
        `got [${actualKeys.join(',')}]. Raw: ${text.slice(0, 200)}`
    }
  }

  return {
    passed: true,
    output: `json_schema OK — Person { name: ${obj.name}, age: ${obj.age}, occupation: ${obj.occupation} }`
  }
}
