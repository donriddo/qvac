import { send } from "@/client/rpc/rpc-client";
import {
  type EmbedParams,
  type EmbedRequest,
  type EmbedStats,
  type RPCOptions,
} from "@/schemas";
import { InvalidResponseError } from "@/utils/errors-client";

export type { EmbedStats };

/**
 * Generates embeddings for a single text using a specified model.
 *
 * @param params - The parameters for the embedding
 * @param params.modelId - The identifier of the embedding model to use
 * @param params.text - The input text to embed
 * @param options - Optional RPC options including per-call profiling
 * @throws {QvacErrorBase} When the response type is invalid or when the embedding fails
 */
export async function embed(
  params: { modelId: string; text: string },
  options?: RPCOptions,
): Promise<number[]>;

/**
 * Generates embeddings for multiple texts using a specified model.
 *
 * @param params - The parameters for the embedding
 * @param params.modelId - The identifier of the embedding model to use
 * @param params.text - The input texts to embed
 * @param options - Optional RPC options including per-call profiling
 * @throws {QvacErrorBase} When the response type is invalid or when the embedding fails
 */
export async function embed(
  params: { modelId: string; text: string[] },
  options?: RPCOptions,
): Promise<number[][]>;

export async function embed(
  params: EmbedParams,
  options?: RPCOptions,
): Promise<number[] | number[][]> {
  const request: EmbedRequest = {
    type: "embed",
    ...params,
  };

  const response = await send(request, options);
  if (response.type !== "embed") {
    throw new InvalidResponseError("embed");
  }

  return response.embedding;
}

/**
 * Generates embeddings together with the addon runtime stats (TPS, total time,
 * resolved backend device, etc.). Use this when you need the stats in addition
 * to the embedding vectors.
 *
 * @param params - The parameters for the embedding
 * @param params.modelId - The identifier of the embedding model to use
 * @param params.text - The input text or texts to embed
 * @param options - Optional RPC options including per-call profiling
 * @throws {QvacErrorBase} When the response type is invalid or when the embedding fails
 */
export async function embedWithStats(
  params: { modelId: string; text: string },
  options?: RPCOptions,
): Promise<{ embedding: number[]; stats?: EmbedStats }>;
export async function embedWithStats(
  params: { modelId: string; text: string[] },
  options?: RPCOptions,
): Promise<{ embedding: number[][]; stats?: EmbedStats }>;
export async function embedWithStats(
  params: EmbedParams,
  options?: RPCOptions,
): Promise<{ embedding: number[] | number[][]; stats?: EmbedStats }> {
  const request: EmbedRequest = {
    type: "embed",
    ...params,
  };

  const response = await send(request, options);
  if (response.type !== "embed") {
    throw new InvalidResponseError("embed");
  }

  return response.stats !== undefined
    ? { embedding: response.embedding, stats: response.stats }
    : { embedding: response.embedding };
}
