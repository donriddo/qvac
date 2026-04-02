import { getModel } from "@/server/bare/registry/model-registry";
import { ttsRequestSchema, type TtsRequest, type TtsStats } from "@/schemas";
import { nowMs } from "@/profiling";
import { buildStreamResult, hasDefinedValues } from "@/profiling/model-execution";
import type { TtsResponse } from "@/server/bare/types/addon-responses";

export async function* textToSpeech(
  params: TtsRequest,
): AsyncGenerator<{ buffer: number[]; sampleRate?: number }, { modelExecutionMs: number; stats?: TtsStats }> {
  const parsed = ttsRequestSchema.parse(params);
  const { modelId, text, stream } = parsed;

  const model = getModel(modelId);

  const runInput: Record<string, unknown> = { input: text, inputType: parsed.inputType };
  if (parsed.outputSampleRate !== undefined) runInput["outputSampleRate"] = parsed.outputSampleRate;
  if (parsed.enhancer?.type === "lavasr") {
    if (parsed.enhancer.enhance !== undefined) runInput["enhance"] = parsed.enhancer.enhance;
    if (parsed.enhancer.denoise !== undefined) runInput["denoise"] = parsed.enhancer.denoise;
  }

  const modelStart = nowMs();
  const response = (await model.run(runInput)) as unknown as TtsResponse;

  let lastSampleRate: number | undefined;

  if (!stream) {
    let completeBuffer: number[] = [];

    for await (const data of response.iterate()) {
      completeBuffer = completeBuffer.concat(Array.from(data.outputArray));
      if (data.sampleRate !== undefined) lastSampleRate = data.sampleRate;
    }

    const modelExecutionMs = nowMs() - modelStart;
    const stats: TtsStats = {
      ...(response.stats?.audioDurationMs !== undefined && { audioDuration: response.stats.audioDurationMs }),
      ...(response.stats?.totalSamples !== undefined && { totalSamples: response.stats.totalSamples }),
      ...(response.stats?.sampleRate !== undefined && { sampleRate: response.stats.sampleRate }),
      ...(lastSampleRate !== undefined && { sampleRate: lastSampleRate }),
    };

    yield { buffer: completeBuffer, ...(lastSampleRate !== undefined && { sampleRate: lastSampleRate }) };
    return buildStreamResult(modelExecutionMs, hasDefinedValues(stats) ? stats : undefined);
  }

  for await (const data of response.iterate()) {
    if (data.sampleRate !== undefined) lastSampleRate = data.sampleRate;
    yield { buffer: Array.from(data.outputArray), ...(data.sampleRate !== undefined && { sampleRate: data.sampleRate }) };
  }

  const modelExecutionMs = nowMs() - modelStart;
  const stats: TtsStats = {
    ...(response.stats?.audioDurationMs !== undefined && { audioDuration: response.stats.audioDurationMs }),
    ...(response.stats?.totalSamples !== undefined && { totalSamples: response.stats.totalSamples }),
    ...(response.stats?.sampleRate !== undefined && { sampleRate: response.stats.sampleRate }),
    ...(lastSampleRate !== undefined && { sampleRate: lastSampleRate }),
  };

  return buildStreamResult(modelExecutionMs, hasDefinedValues(stats) ? stats : undefined);
}
