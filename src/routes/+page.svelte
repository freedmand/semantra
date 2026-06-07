<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import FileEmbedder from "$lib/embedding/FileEmbedder.svelte";

  // Sensible default: cap how many items go to the model per call. Batching does
  // NOT change results (pooling/normalization are per-row and padding is masked),
  // so this is purely a memory/throughput guard for long lists.
  const BATCH_SIZE = 32;

  let itemsText = $state(
    [
      "Machine learning is a subset of artificial intelligence.",
      "Neural networks are trained through backpropagation.",
      "The striker scored a last-minute goal to win the match.",
      "She broke the national record in the sprint.",
      "Simmer the tomato sauce on low heat to deepen the flavor.",
    ].join("\n"),
  );
  // Optional: prefix every item with the asymmetric query prefix. Off = symmetric
  // document-vs-document comparison, which is the right default for a self-matrix.
  let asQueries = $state(false);
  let status = $state("");
  let busy = $state(false);
  let labels = $state<string[]>([]);
  let matrix = $state<number[][]>([]);

  function parseItems(text: string): string[] {
    return text
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  // Embed a list in batches of BATCH_SIZE, concatenating the rows.
  async function embedBatched(
    items: string[],
    isQuery: boolean,
  ): Promise<number[][]> {
    const rows: number[][] = [];
    for (let i = 0; i < items.length; i += BATCH_SIZE) {
      const chunk = items.slice(i, i + BATCH_SIZE);
      // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
      const embs = await invoke<number[][]>("embed", {
        texts: chunk,
        isQuery,
      });
      rows.push(...embs);
      status = `embedding ${Math.min(i + BATCH_SIZE, items.length)} / ${items.length}…`;
    }
    return rows;
  }

  // Rows are unit-norm, so dot product == cosine similarity.
  function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  async function compare(event: Event) {
    event.preventDefault();
    if (busy) return;
    const items = parseItems(itemsText);
    if (items.length < 2) {
      status = "enter at least two items (one per line)";
      matrix = [];
      labels = [];
      return;
    }
    busy = true;
    try {
      status = "embedding…";
      // Wall-clock for the embedding round-trips — this is the "query time".
      const t0 = performance.now();
      const embs = await embedBatched(items, asQueries);
      const elapsedMs = performance.now() - t0;
      // full pairwise similarity matrix
      matrix = embs.map((a) => embs.map((b) => dot(a, b)));
      labels = items;
      const time =
        elapsedMs < 1000
          ? `${elapsedMs.toFixed(0)} ms`
          : `${(elapsedMs / 1000).toFixed(2)} s`;
      const batches = Math.ceil(items.length / BATCH_SIZE);
      status = `${items.length} items · ${batches} ${batches === 1 ? "batch" : "batches"} of ${BATCH_SIZE} · ${time}`;
    } catch (e) {
      status = `error: ${e}`;
      matrix = [];
      labels = [];
    } finally {
      busy = false;
    }
  }

  // Map a similarity to a sequential green background + readable text color.
  // Clamped to [0,1]; the diagonal (self-similarity = 1.0) anchors the darkest.
  function cellStyle(sim: number): string {
    const t = Math.max(0, Math.min(1, sim));
    const lightness = 96 - t * 52; // 96% (low) -> 44% (high)
    const text = lightness < 62 ? "#fff" : "#0f0f0f";
    return `background: hsl(145 58% ${lightness}%); color: ${text};`;
  }

  function truncate(s: string, n = 32): string {
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }
</script>

<main class="container">
  <h1>mdbr-leaf-ir similarity matrix</h1>
  <p>One item per line — every pair is embedded in Rust and scored by cosine similarity.</p>

  <form class="controls" onsubmit={compare}>
    <textarea
      rows="6"
      placeholder="One item per line…"
      bind:value={itemsText}
    ></textarea>
    <div class="row">
      <label class="checkbox">
        <input type="checkbox" bind:checked={asQueries} />
        treat items as search queries
      </label>
      <button type="submit" disabled={busy}>
        {busy ? "Comparing…" : "Compare"}
      </button>
    </div>
  </form>

  {#if status}<p class="status">{status}</p>{/if}

  {#if matrix.length}
    <div class="matrix-wrap">
      <table class="matrix">
        <thead>
          <tr>
            <th class="corner"></th>
            {#each labels as label, j}
              <th class="col-head" title={label}>{j + 1}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each matrix as rowSims, i}
            <tr>
              <th class="row-head" title={labels[i]}>
                {i + 1}. {truncate(labels[i])}
              </th>
              {#each rowSims as sim}
                <td style={cellStyle(sim)}>{sim.toFixed(2)}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <FileEmbedder />
</main>

<style>
:root {
  font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 24px;
  font-weight: 400;

  color: #0f0f0f;
  background-color: #f6f6f6;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}

.container {
  margin: 0 auto;
  max-width: 1000px;
  padding: 6vh 1rem 4rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  width: 100%;
  max-width: 640px;
}

.row {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

.checkbox {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.9em;
}

.status {
  color: #666;
  font-size: 0.9em;
}

h1 {
  text-align: center;
}

textarea {
  width: 100%;
  resize: vertical;
  border-radius: 8px;
  border: 1px solid #ddd;
  padding: 0.6em 0.8em;
  font-size: 1em;
  font-family: inherit;
  color: #0f0f0f;
  background-color: #ffffff;
  box-shadow: 0 2px 2px rgba(0, 0, 0, 0.1);
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
button:active:not(:disabled) {
  border-color: #396cd8;
  background-color: #e8e8e8;
}
button:disabled {
  cursor: default;
  opacity: 0.6;
}

textarea,
button,
input {
  outline: none;
}

.matrix-wrap {
  margin-top: 1.5rem;
  width: 100%;
  overflow: auto;
}

.matrix {
  border-collapse: collapse;
  margin: 0 auto;
  font-size: 0.85em;
}

.matrix th,
.matrix td {
  border: 1px solid #e2e2e2;
  padding: 0.35em 0.5em;
}

.matrix td {
  text-align: center;
  font-variant-numeric: tabular-nums;
  min-width: 2.6em;
}

.col-head {
  font-weight: 600;
}

.row-head {
  text-align: left;
  white-space: nowrap;
  font-weight: 500;
  max-width: 22em;
  overflow: hidden;
  text-overflow: ellipsis;
}

.corner {
  background: transparent;
  border: none;
}

@media (prefers-color-scheme: dark) {
  :root {
    color: #f6f6f6;
    background-color: #2f2f2f;
  }

  .status {
    color: #aaa;
  }

  textarea,
  button {
    color: #ffffff;
    background-color: #0f0f0f98;
    border-color: #444;
  }
  button:active:not(:disabled) {
    background-color: #0f0f0f69;
  }

  .matrix th,
  .matrix td {
    border-color: #555;
  }
}

</style>
