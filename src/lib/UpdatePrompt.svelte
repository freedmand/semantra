<script lang="ts">
  // In-app updates: on launch, ask GitHub Releases (the `latest.json` endpoint
  // in tauri.conf.json) for a newer signed build. If one exists, offer to
  // download, install and relaunch. Offline / dev / no-update are all silent.
  import { onMount } from "svelte";
  import { check, type Update } from "@tauri-apps/plugin-updater";
  import { relaunch } from "@tauri-apps/plugin-process";

  let update: Update | null = $state(null);
  let dismissed = $state(false);
  let installing = $state(false);
  let progress: number | null = $state(null); // 0–1, null if size unknown
  let error: string | null = $state(null);

  onMount(async () => {
    if (import.meta.env.DEV) return;
    try {
      update = await check();
    } catch (e) {
      console.warn("update check failed", e);
    }
  });

  async function install() {
    if (!update) return;
    installing = true;
    error = null;
    let total = 0;
    let done = 0;
    try {
      await update.downloadAndInstall((ev) => {
        if (ev.event === "Started") total = ev.data.contentLength ?? 0;
        else if (ev.event === "Progress") {
          done += ev.data.chunkLength;
          progress = total ? done / total : null;
        }
      });
      await relaunch();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      installing = false;
    }
  }
</script>

{#if update && !dismissed}
  <div class="update" role="status">
    <div class="msg">
      <strong>Semantra {update.version} is available</strong>
      {#if installing}
        <span class="dim">
          {progress == null ? "Downloading…" : `Downloading… ${Math.round(progress * 100)}%`}
        </span>
      {:else if error}
        <span class="err">Update failed: {error}</span>
      {:else}
        <span class="dim">You have {update.currentVersion}.</span>
      {/if}
    </div>
    <div class="actions">
      {#if !installing}
        <button class="later" onclick={() => (dismissed = true)}>Later</button>
        <button class="primary" onclick={install}>{error ? "Retry" : "Install and restart"}</button>
      {/if}
    </div>
  </div>
{/if}

<style>
  .update {
    position: fixed;
    right: 16px;
    bottom: 16px;
    z-index: 100;
    display: flex;
    gap: 16px;
    align-items: center;
    max-width: 440px;
    padding: 12px 14px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    box-shadow: 0 6px 24px rgb(0 0 0 / 15%);
    font-size: 0.9rem;
  }
  .msg {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .dim {
    color: var(--color-text-muted);
  }
  .err {
    color: var(--color-error);
  }
  .actions {
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }
  button {
    padding: 5px 10px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
    background: white;
    color: var(--color-text);
    cursor: pointer;
  }
  button:hover {
    background: var(--color-bg-hover);
  }
  .primary {
    background: var(--color-accent);
    border-color: var(--color-accent);
    color: white;
  }
  .primary:hover {
    background: var(--color-accent);
    filter: brightness(1.1);
  }
</style>
