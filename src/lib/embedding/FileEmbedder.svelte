<script lang="ts">
  // Self-contained file -> chunk -> embed widget. Drop it anywhere; it owns its
  // own state and talks to the backend through the shared embedding client, so
  // the upcoming file-upload flow can reuse it without change.
  import { chunkWords } from "./chunk";
  import { embedChunks, type EmbedFinished } from "./embed";

  // Words per chunk. Exposed as a prop so callers can override the default.
  let { chunkSize = 128 }: { chunkSize?: number } = $props();

  let fileName = $state("");
  let chunks = $state<string[]>([]);
  let busy = $state(false);
  let error = $state("");

  // Live progress, updated from streamed events.
  let total = $state(0);
  let done = $state(0);
  let batchIndex = $state(0);
  let totalBatches = $state(0);
  let lastBatchMs = $state(0);
  let elapsedMs = $state(0);
  let summary = $state<EmbedFinished | null>(null);

  // Fraction complete (0..1) for the progress bar.
  const fraction = $derived(total > 0 ? done / total : 0);
  // Rough ETA: extrapolate remaining work from the average per-chunk time so far.
  const etaMs = $derived(
    done > 0 && done < total ? (elapsedMs / done) * (total - done) : 0,
  );

  // Format a millisecond duration the way the rest of the app does.
  function fmt(ms: number): string {
    return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`;
  }

  async function onFile(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    fileName = file.name;
    error = "";
    summary = null;
    try {
      const text = await file.text();
      chunks = chunkWords(text, chunkSize);
    } catch (e) {
      error = `could not read file: ${e}`;
      chunks = [];
    }
  }

  async function run() {
    if (busy || chunks.length === 0) return;
    busy = true;
    error = "";
    summary = null;
    total = chunks.length;
    done = 0;
    batchIndex = 0;
    totalBatches = 0;
    lastBatchMs = 0;
    elapsedMs = 0;
    try {
      await embedChunks(chunks, false, (p) => {
        switch (p.kind) {
          case "started":
            total = p.totalChunks;
            totalBatches = p.totalBatches;
            break;
          case "batch":
            done = p.chunksDone;
            batchIndex = p.batchIndex + 1; // 1-based for display
            totalBatches = p.totalBatches;
            lastBatchMs = p.batchMs;
            elapsedMs = p.elapsedMs;
            break;
          case "finished":
            done = p.totalChunks;
            elapsedMs = p.totalMs;
            summary = p;
            break;
        }
      });
    } catch (e) {
      error = `embedding failed: ${e}`;
    } finally {
      busy = false;
    }
  }
</script>

<section class="embedder">
  <h2>Embed a text file</h2>
  <p>
    Pick a <code>.txt</code> file — it's split into {chunkSize}-word chunks and
    embedded in Rust. The vectors are discarded; this measures the pipeline.
  </p>

  <div class="row">
    <input
      type="file"
      accept=".txt,text/plain"
      onchange={onFile}
      disabled={busy}
    />
    <button type="button" onclick={run} disabled={busy || chunks.length === 0}>
      {busy ? "Embedding…" : "Embed file"}
    </button>
  </div>

  {#if fileName}
    <p class="status">
      <strong>{fileName}</strong> · {chunks.length}
      {chunks.length === 1 ? "chunk" : "chunks"}
    </p>
  {/if}

  {#if error}<p class="status error">{error}</p>{/if}

  {#if busy || summary}
    <div class="progress-wrap">
      <progress max={total || 1} value={done}></progress>
      <p class="status timing">
        {done} / {total} chunks
        {#if totalBatches}· batch {batchIndex}/{totalBatches}{/if}
        {#if lastBatchMs}· last {fmt(lastBatchMs)}{/if}
        · elapsed {fmt(elapsedMs)}
        {#if etaMs > 0}· ETA ~{fmt(etaMs)}{/if}
      </p>
    </div>
  {/if}

  {#if summary}
    <dl class="summary">
      <div><dt>Total time</dt><dd>{fmt(summary.totalMs)}</dd></div>
      <div>
        <dt>Chunks · batches</dt>
        <dd>{summary.totalChunks} · {summary.totalBatches}</dd>
      </div>
      <div>
        <dt>Throughput</dt>
        <dd>{summary.chunksPerSec.toFixed(1)} chunks/s</dd>
      </div>
      <div>
        <dt>Batch avg / min / max</dt>
        <dd>
          {fmt(summary.avgBatchMs)} / {fmt(summary.minBatchMs)} / {fmt(
            summary.maxBatchMs,
          )}
        </dd>
      </div>
    </dl>
  {/if}
</section>

<style>
  .embedder {
    width: 100%;
    max-width: 640px;
    margin: 2.5rem auto 0;
    border-top: 1px solid #ddd;
    padding-top: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    align-items: center;
    text-align: center;
  }

  .row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .status {
    color: #666;
    font-size: 0.9em;
    margin: 0;
  }

  .status.error {
    color: #c0392b;
  }

  .progress-wrap {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    align-items: center;
  }

  progress {
    width: 100%;
    height: 0.7rem;
    accent-color: hsl(145 58% 44%);
  }

  .timing {
    font-variant-numeric: tabular-nums;
  }

  .summary {
    width: 100%;
    margin: 0.5rem 0 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem 1rem;
    font-size: 0.9em;
  }

  .summary div {
    border: 1px solid #e2e2e2;
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    text-align: left;
  }

  .summary dt {
    color: #888;
    font-size: 0.85em;
  }

  .summary dd {
    margin: 0.15rem 0 0;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }

  button {
    border-radius: 8px;
    border: 1px solid transparent;
    padding: 0.6em 1.2em;
    font-size: 1em;
    font-weight: 500;
    font-family: inherit;
    color: #0f0f0f;
    background-color: #ffffff;
    transition: border-color 0.25s;
    box-shadow: 0 2px 2px rgba(0, 0, 0, 0.2);
    cursor: pointer;
  }

  button:hover:not(:disabled) {
    border-color: #396cd8;
  }
  button:disabled {
    cursor: default;
    opacity: 0.6;
  }

  @media (prefers-color-scheme: dark) {
    .embedder {
      border-color: #444;
    }
    .status {
      color: #aaa;
    }
    .summary div {
      border-color: #555;
    }
    button {
      color: #ffffff;
      background-color: #0f0f0f98;
      border-color: #444;
    }
  }
</style>
