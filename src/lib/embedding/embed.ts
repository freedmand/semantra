/**
 * Frontend client for the streaming `embed_chunks` Tauri command.
 *
 * Mirrors the Rust `EmbedProgress` enum (see `src-tauri/src/embed.rs`) and wraps
 * the `Channel`-based invoke so callers just provide an `onProgress` handler.
 * Kept free of UI concerns so both this demo and the upcoming file-upload
 * feature can share it.
 */
import { Channel, invoke } from "@tauri-apps/api/core";

/** Sent once before any work begins. */
export interface EmbedStarted {
  kind: "started";
  totalChunks: number;
  totalBatches: number;
  batchSize: number;
}

/** Sent after each batch completes. Timings are in milliseconds. */
export interface EmbedBatch {
  kind: "batch";
  batchIndex: number;
  totalBatches: number;
  chunksDone: number;
  totalChunks: number;
  batchMs: number;
  elapsedMs: number;
}

/** Sent once after the final batch, with summary timing stats. */
export interface EmbedFinished {
  kind: "finished";
  totalChunks: number;
  totalBatches: number;
  totalMs: number;
  chunksPerSec: number;
  avgBatchMs: number;
  minBatchMs: number;
  maxBatchMs: number;
}

/** Discriminated union of every progress event, keyed on `kind`. */
export type EmbedProgress = EmbedStarted | EmbedBatch | EmbedFinished;

/**
 * Embed `chunks` in the Rust backend, discarding the vectors, streaming
 * progress to `onProgress`. Resolves once the backend has emitted `finished`.
 *
 * @param chunks      the text chunks to embed
 * @param isQuery     prefix each chunk with the asymmetric query prefix
 * @param onProgress  called for every {@link EmbedProgress} event, in order
 */
export async function embedChunks(
  chunks: string[],
  isQuery: boolean,
  onProgress: (p: EmbedProgress) => void,
): Promise<void> {
  const channel = new Channel<EmbedProgress>();
  channel.onmessage = onProgress;
  await invoke("embed_chunks", { chunks, isQuery, onProgress: channel });
}
