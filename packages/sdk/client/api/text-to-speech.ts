import {
  ttsResponseSchema,
  type TtsClientParams,
  type TtsRequest,
  type RPCOptions,
} from "@/schemas";
import { stream as streamRpc } from "@/client/rpc/rpc-client";

export function textToSpeech(
  params: TtsClientParams,
  options?: RPCOptions,
): {
  bufferStream: AsyncGenerator<number>;
  buffer: Promise<number[]>;
  done: Promise<boolean>;
  sampleRate: Promise<number | undefined>;
} {
  const request: TtsRequest = {
    type: "textToSpeech",
    modelId: params.modelId,
    inputType: params.inputType,
    text: params.text,
    stream: params.stream,
    ...(params.outputSampleRate !== undefined && { outputSampleRate: params.outputSampleRate }),
    ...(params.enhancer !== undefined && { enhancer: params.enhancer }),
  };

  let doneResolver: (value: boolean) => void = () => {};
  const donePromise = new Promise<boolean>((resolve) => {
    doneResolver = resolve;
  });

  let sampleRateResolver: (value: number | undefined) => void = () => {};
  const sampleRatePromise = new Promise<number | undefined>((resolve) => {
    sampleRateResolver = resolve;
  });

  if (params.stream) {
    const bufferStream = (async function* () {
      let lastSampleRate: number | undefined;
      for await (const response of streamRpc(request, options)) {
        if (response.type === "textToSpeech") {
          const streamResponse = ttsResponseSchema.parse(response);
          if (streamResponse.sampleRate !== undefined) lastSampleRate = streamResponse.sampleRate;
          if (streamResponse.buffer.length > 0) {
            yield* streamResponse.buffer;
          }
          if (streamResponse.done) {
            sampleRateResolver(lastSampleRate ?? streamResponse.stats?.sampleRate);
            doneResolver(true);
          }
        }
      }
    })();

    return {
      bufferStream,
      buffer: Promise.resolve([]),
      done: donePromise,
      sampleRate: sampleRatePromise,
    };
  } else {
    const bufferStream = (async function* () {
      //Empty generator for non-streaming mode
    })();

    const bufferPromise = (async () => {
      let buffer: number[] = [];
      let lastSampleRate: number | undefined;
      for await (const response of streamRpc(request, options)) {
        if (response.type === "textToSpeech") {
          const streamResponse = ttsResponseSchema.parse(response);
          if (streamResponse.sampleRate !== undefined) lastSampleRate = streamResponse.sampleRate;
          buffer = buffer.concat(streamResponse.buffer);
          if (streamResponse.done) {
            sampleRateResolver(lastSampleRate ?? streamResponse.stats?.sampleRate);
            doneResolver(true);
          }
        }
      }
      return buffer;
    })();

    return {
      bufferStream,
      buffer: bufferPromise,
      done: donePromise,
      sampleRate: sampleRatePromise,
    };
  }
}
