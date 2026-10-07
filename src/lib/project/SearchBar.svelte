<script lang="ts">
  // Query input + attachments + relevance-feedback preference chips. Reads and
  // writes the central search state directly. Flags the query as "outdated"
  // (dashed yellow) when it has changed since the last search — that staleness
  // tracking is local UI.
  //
  // Two subtle icon buttons sit inside the input: add an image/audio file, and
  // record a voice clip. Attachments are embedded together with the query text
  // as one interleaved query (the model composes modalities natively). Files
  // dropped onto the search page attach too.
  import { onMount, onDestroy } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import type { UnlistenFn } from "@tauri-apps/api/event";
  import {
    appState,
    runSearch,
    setPreference,
    addAttachment,
    removeAttachment,
  } from "$lib/state.svelte";
  import { formatTime, thumbnailForPath } from "./projectClient";

  const search = appState.search;

  const IMAGE_EXTS = ["jpg", "jpeg", "png", "gif", "webp", "heic", "heif", "avif", "bmp", "tif", "tiff"];
  const AUDIO_EXTS = ["mp3", "m4a", "aac", "wav", "aif", "aiff", "caf", "flac"];
  const kindOf = (path: string): "image" | "audio" | null => {
    const ext = path.split(".").pop()?.toLowerCase() ?? "";
    return IMAGE_EXTS.includes(ext) ? "image" : AUDIO_EXTS.includes(ext) ? "audio" : null;
  };
  const baseName = (path: string) => path.split("/").pop() ?? path;

  // Image chip thumbnails, by attachment id.
  let thumbs = $state<Record<number, string>>({});

  async function attachPaths(paths: string[]) {
    for (const path of paths) {
      const kind = kindOf(path);
      if (!kind) continue;
      await addAttachment({ kind, path, label: baseName(path) });
      const a = search.attachments[search.attachments.length - 1];
      if (kind === "image" && a) {
        thumbnailForPath(path).then((t) => (thumbs = { ...thumbs, [a.id]: t })).catch(() => {});
      }
    }
    lastKey = searchKey;
  }

  async function pickFiles() {
    const selected = await open({
      multiple: true,
      directory: false,
      filters: [{ name: "Images & audio", extensions: [...IMAGE_EXTS, ...AUDIO_EXTS] }],
    });
    const paths = Array.isArray(selected) ? selected : selected ? [selected] : [];
    if (paths.length) await attachPaths(paths);
  }

  // --- Voice recording ------------------------------------------------------
  let recorder: MediaRecorder | null = null;
  let recording = $state(false);
  let recStart = 0;
  let recElapsed = $state(0);
  let recTimer: ReturnType<typeof setInterval> | null = null;
  let micError = $state("");
  const MAX_RECORDING_MS = 30_000; // the model embeds at most 30 s of audio

  async function toggleRecording() {
    if (recording) {
      recorder?.stop();
      return;
    }
    micError = "";
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      micError = "Microphone unavailable";
      return;
    }
    const mimeType = ["audio/mp4", "audio/webm"].find((t) => MediaRecorder.isTypeSupported(t)) ?? "";
    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      if (recTimer) clearInterval(recTimer);
      recording = false;
      const ms = Date.now() - recStart;
      const blob = new Blob(chunks, { type: recorder?.mimeType || mimeType });
      if (ms < 300 || blob.size === 0) return;
      const ext = blob.type.includes("webm") ? "webm" : "m4a";
      const dataBase64 = await blobToBase64(blob);
      await addAttachment({ kind: "audio", dataBase64, ext, label: `Recording ${formatTime(ms)}` });
      lastKey = searchKey;
    };
    recorder.start();
    recording = true;
    recStart = Date.now();
    recElapsed = 0;
    recTimer = setInterval(() => {
      recElapsed = Date.now() - recStart;
      if (recElapsed >= MAX_RECORDING_MS) recorder?.stop();
    }, 200);
  }

  function blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result).split(",")[1] ?? "");
      r.onerror = () => reject(r.error);
      r.readAsDataURL(blob);
    });
  }

  // Files dropped anywhere on the search page attach to the query.
  let unlisten: UnlistenFn | null = null;
  onMount(async () => {
    unlisten = await getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "drop") attachPaths(event.payload.paths);
    });
  });
  onDestroy(() => {
    unlisten?.();
    if (recording) recorder?.stop();
  });

  // A chip's strength is the magnitude of its signed weight; the sign carries
  // the +/- direction. Clamp to the input's 0.1–2.0 range.
  function clampMultiple(v: number): number {
    if (Number.isNaN(v)) return 1;
    return Math.min(2, Math.max(0.1, v));
  }

  // Track the (query, preferences, indexed-doc-count) state at the last search to
  // show staleness — including doc count so the bar re-yellows when more files
  // finish indexing after a search (their results aren't reflected yet).
  let lastKey = $state("");
  const searchKey = $derived(
    JSON.stringify({
      value: search.query,
      prefs: appState.prefList.map((p) => [p.hit.index, p.weight]),
      attachments: search.attachments.map((a) => a.id),
      docs: search.docs.length,
    }),
  );
  const outdated = $derived(searchKey !== lastKey);

  function doSearch() {
    runSearch(search.query);
    lastKey = searchKey;
  }
</script>

<div class="flex flex-1 flex-col">
  <div class="flex items-center relative flex-1">
    <input
      class="bg-white py-2 px-4 pl-12 pr-20 font-mono w-full rounded-sm border"
      class:outdated
      style="color:#0f0f0f; border-color: var(--color-border);"
      placeholder="Search"
      autocorrect="off"
      autocapitalize="off"
      autocomplete="off"
      spellcheck="false"
      bind:value={search.query}
      onkeydown={(e) => {
        if (e.key === "Enter") doSearch();
      }}
    />
    <button class="search-button" onclick={doSearch} aria-label="Search">Search</button>
    <div class="input-tools">
      <button class="tool" onclick={pickFiles} title="Search with an image or audio file" aria-label="Add image or audio">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="#202020" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"
          ><rect x="2.5" y="3.5" width="12" height="13" rx="1.5" /><path d="M5 13l2.6-3 2 2.2 1.4-1.6 1.5 2.4" /><circle cx="7" cy="7.2" r="1.1" /><path d="M16.5 1.8v5M14 4.3h5" /></svg
        >
      </button>
      <button
        class="tool"
        class:recording
        onclick={toggleRecording}
        title={recording ? "Stop recording" : "Search by voice"}
        aria-label={recording ? "Stop recording" : "Record voice search"}
        aria-pressed={recording}
      >
        {#if recording}
          <span class="rec-time">{formatTime(recElapsed)}</span>
        {/if}
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke={recording ? "#c0392b" : "#202020"} stroke-width="1.6" stroke-linecap="round"
          ><rect x="7" y="2.5" width="6" height="10" rx="3" fill={recording ? "#c0392b" : "none"} /><path d="M4.5 9.5a5.5 5.5 0 0 0 11 0M10 15v2.5" /></svg
        >
      </button>
    </div>
  </div>

  {#if search.attachments.length || micError}
    <div class="mt-2 flex flex-wrap gap-2 items-center">
      {#each search.attachments as a (a.id)}
        <div class="attachment" title={a.path ?? a.label}>
          {#if a.kind === "image" && thumbs[a.id]}
            <img src={thumbs[a.id]} alt="" />
          {:else}
            <span class="att-icon" aria-hidden="true">{a.kind === "image" ? "▣" : "♪"}</span>
          {/if}
          <span class="truncate max-w-40">{a.label}</span>
          <button class="att-remove" onclick={() => removeAttachment(a.id)} aria-label={`Remove ${a.label}`}>×</button>
        </div>
      {/each}
      {#if micError}
        <span class="text-xs" style="color: var(--color-error);">{micError}</span>
      {/if}
    </div>
  {/if}

  {#if appState.prefList.length}
    <div class="max-h-24 overflow-y-auto mt-2 flex flex-wrap gap-2">
      {#each appState.prefList as pref (pref.hit.index)}
        <div
          class="flex items-center font-mono rounded-sm text-sm max-w-72"
          style={pref.weight > 0 ? "background:#bfdbfe;" : "background:#fed7aa;"}
        >
          <button
            class="max-w-48 truncate px-2 py-0.5"
            title={`${pref.hit.basename}: ${pref.hit.text} (click to remove)`}
            onclick={() => setPreference(pref.hit, 0)}
          >
            <span
              class="font-bold mr-1"
              style={pref.weight > 0 ? "color:#2563eb;" : "color:#f97316;"}
              >{pref.weight > 0 ? "+" : "-"}</span
            >{pref.hit.text}
          </button>
          <input
            type="number"
            class="w-12 rounded-sm border bg-white/70 px-1 py-0.5 mr-1 text-xs"
            style="color:#0f0f0f; border-color: var(--color-border);"
            min="0.1"
            max="2"
            step="0.1"
            title="Strength multiplier (0.1–2.0)"
            value={Math.abs(pref.weight)}
            onclick={(e) => e.stopPropagation()}
            onchange={(e) => {
              const v = clampMultiple((e.currentTarget as HTMLInputElement).valueAsNumber);
              setPreference(pref.hit, Math.sign(pref.weight) * v);
            }}
          />
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .outdated {
    background: #fefce8;
    border-style: dashed !important;
    border-color: #ca8a04 !important;
  }

  .input-tools {
    position: absolute;
    right: 8px;
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .tool {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 28px;
    padding: 0 4px;
    border-radius: var(--radius-sm);
    opacity: 0.55;
  }
  .tool:hover,
  .tool.recording {
    opacity: 1;
    background: var(--color-bg-hover);
  }
  .rec-time {
    font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
    color: #c0392b;
  }
  .attachment {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 28px;
    padding: 0 4px 0 4px;
    font-size: 0.8rem;
    border: 1px solid var(--color-border-soft);
    border-radius: var(--radius-sm);
    background: var(--color-bg-elevated);
  }
  .attachment img {
    width: 22px;
    height: 22px;
    object-fit: cover;
    border-radius: 2px;
  }
  .att-icon {
    color: var(--color-text-muted);
    padding-left: 2px;
  }
  .att-remove {
    width: 18px;
    height: 18px;
    line-height: 1;
    border-radius: var(--radius-sm);
    color: var(--color-text-muted);
  }
  .att-remove:hover {
    background: var(--color-bg-hover);
    color: var(--color-text);
  }

  .search-button {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='25' height='24' fill='none'%3E%3Cpath stroke='%23202020' stroke-width='3' d='M10.045 13.424A7.152 7.152 0 1 0 21.003 4.23a7.152 7.152 0 0 0-10.958 9.194Zm0 0-8.984 8.984'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    text-indent: -9999px;
    width: 25px;
    height: 24px;
    position: absolute;
    left: 12px;
  }
</style>
