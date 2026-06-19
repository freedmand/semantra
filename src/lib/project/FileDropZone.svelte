<script lang="ts">
  // A visible, clickable file dropzone: a big dashed rounded rectangle that
  // opens the file picker on click and accepts dropped files. Tauri intercepts
  // OS file drops natively (the window's dragDropEnabled default), so we listen
  // to the webview drag-drop event — which carries real filesystem paths —
  // rather than HTML5 drop events. Both the click-picker and the drop emit the
  // chosen paths through `onFiles`.
  import { onMount, onDestroy } from "svelte";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { open } from "@tauri-apps/plugin-dialog";
  import type { UnlistenFn } from "@tauri-apps/api/event";

  let {
    onFiles,
    label = "Drag files here, or click to upload",
    busyLabel = "Adding…",
    busy = false,
  }: {
    onFiles: (paths: string[]) => void;
    label?: string;
    busyLabel?: string;
    busy?: boolean;
  } = $props();

  let dragging = $state(false);
  let unlisten: UnlistenFn | null = null;

  onMount(async () => {
    unlisten = await getCurrentWebview().onDragDropEvent((event) => {
      const p = event.payload;
      if (p.type === "enter" || p.type === "over") {
        dragging = true;
      } else if (p.type === "leave") {
        dragging = false;
      } else if (p.type === "drop") {
        dragging = false;
        if (p.paths.length) onFiles(p.paths);
      }
    });
  });
  onDestroy(() => unlisten?.());

  async function pick() {
    const selected = await open({
      multiple: true,
      directory: false,
      filters: [
        {
          name: "Documents",
          extensions: ["pdf", "txt", "md", "markdown", "text", "csv", "tsv", "log", "json"],
        },
        { name: "All Files", extensions: ["*"] },
      ],
    });
    const paths = Array.isArray(selected) ? selected : selected ? [selected] : [];
    if (paths.length) onFiles(paths);
  }
</script>

<button type="button" class="dropzone" class:dragging onclick={pick} disabled={busy}>
  <svg
    width="28"
    height="28"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path d="M12 16V4" />
    <path d="m6 10 6-6 6 6" />
    <path d="M4 20h16" />
  </svg>
  <span class="dz-title">{busy ? busyLabel : label}</span>
  <span class="dz-sub">PDFs, text, markdown, CSV, and more</span>
</button>

<style>
  .dropzone {
    width: 100%;
    border: 2px dashed var(--color-border-soft);
    border-radius: 12px;
    background: transparent;
    color: var(--color-text-muted);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 3rem 1.5rem;
    cursor: pointer;
    transition:
      border-color 0.15s ease,
      background 0.15s ease;
  }
  .dropzone:hover:not(:disabled) {
    border-color: var(--color-border-hover);
  }
  .dropzone.dragging {
    border-color: var(--color-accent);
    background: var(--color-accent-muted);
  }
  .dropzone:disabled {
    cursor: default;
    opacity: 0.7;
  }
  .dz-title {
    font-weight: 600;
    color: var(--color-text);
  }
  .dz-sub {
    font-size: 0.75rem;
  }
</style>
