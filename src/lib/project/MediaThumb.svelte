<script lang="ts">
  // Preview for a non-text hit in the results list: the image itself, the
  // matched PDF page, or the video frame at the window start — plus a label
  // saying what matched (page number / time range). Audio hits get a compact
  // time-range badge (the player's waveform does the rest).
  import { formatTime, getThumbnail, type ProjectHit } from "./projectClient";

  let { hit }: { hit: ProjectHit } = $props();

  const label = $derived.by(() => {
    if (hit.filetype === "pdf" && hit.modality === "image") return `Page ${(hit.page ?? 0) + 1} · visual match`;
    if (hit.timeStartMs != null && hit.timeEndMs != null) {
      // Round the end up so sub-second clips don't read "0:00–0:00".
      const span = `${formatTime(hit.timeStartMs)}–${formatTime(Math.ceil(hit.timeEndMs / 1000) * 1000)}`;
      if (hit.modality === "audio") return `${span} · ${hit.filetype === "video" ? "soundtrack" : "audio"}`;
      return `${span} · visual`;
    }
    return "Image";
  });

  const thumb = $derived.by((): Promise<string> | null => {
    if (hit.modality === "audio") return null;
    if (hit.filetype === "pdf") return getThumbnail(hit.sha512, { page: hit.page ?? 0, maxSide: 360 });
    if (hit.filetype === "video") return getThumbnail(hit.sha512, { timeMs: hit.timeStartMs ?? 0, maxSide: 360 });
    return getThumbnail(hit.sha512, { maxSide: 360 });
  });
</script>

<div class="media-hit">
  {#if thumb}
    {#await thumb}
      <div class="ph"></div>
    {:then url}
      <img src={url} alt="" class="thumb" />
    {:catch}
      <div class="ph"></div>
    {/await}
  {:else}
    <div class="audio-badge" aria-hidden="true">
      <svg viewBox="0 0 40 16" width="40" height="16">
        {#each [3, 9, 5, 12, 7, 14, 6, 10, 4, 8] as h, i}
          <rect x={i * 4} y={8 - h / 2} width="2.5" height={h} rx="1" fill="currentColor" />
        {/each}
      </svg>
    </div>
  {/if}
  <div class="label">{label}</div>
</div>

<style>
  .media-hit {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .thumb,
  .ph {
    max-width: 100%;
    max-height: 180px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border-soft);
    object-fit: contain;
    align-self: flex-start;
    background: var(--color-bg-elevated);
  }
  .ph {
    width: 160px;
    height: 100px;
  }
  .audio-badge {
    color: #a16207;
  }
  .label {
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }
</style>
