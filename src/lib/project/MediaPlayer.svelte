<script lang="ts">
  // Audio/video reader. The media element plays the file straight from the
  // app-data asset URL; below it a timeline shows the soundtrack's waveform,
  // every search hit in this file as a marker (audio windows above the
  // waveform midline, video-frame windows below, each lane labelled), and the
  // playhead. Clicking or
  // dragging the timeline seeks; clicking a marker plays that window.
  // `navigate(startMs, endMs)` (from a result click) seeks to the window, plays
  // it, and keeps it highlighted.
  import { appState } from "$lib/state.svelte";
  import { formatTime, getFileUrl, getWaveform, type ProjectHit } from "./projectClient";

  let { sha512, kind }: { sha512: string; kind: "audio" | "video" } = $props();

  let media = $state<HTMLMediaElement | null>(null);
  let src = $state<string | null>(null);
  let error = $state<string | null>(null);
  let duration = $state(0);
  let current = $state(0);
  let playing = $state(false);
  let peaks = $state<number[]>([]);
  let active = $state<{ start: number; end: number } | null>(null);
  let canvas = $state<HTMLCanvasElement | null>(null);
  let width = $state(0);
  let scrubbing = false;

  const BUCKETS = 1600;
  const TIMELINE_H = 72;

  $effect(() => {
    let cancelled = false;
    getFileUrl(sha512)
      .then((u) => !cancelled && (src = u))
      .catch((e) => !cancelled && (error = `${e}`));
    getWaveform(sha512, BUCKETS)
      .then((p) => !cancelled && (peaks = p))
      .catch(() => {});
    return () => (cancelled = true);
  });

  /** Hits in this file that carry a time span (audio + video windows). */
  const markers = $derived(
    appState.search.results.filter(
      (h: ProjectHit) => h.sha512 === sha512 && h.timeStartMs != null && h.timeEndMs != null,
    ),
  );
  const best = $derived(Math.max(1e-6, ...markers.map((m) => m.score)));

  // Redraw the waveform + markers whenever their inputs change.
  $effect(() => {
    const c = canvas;
    if (!c || !width) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(TIMELINE_H * dpr);
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, width, TIMELINE_H);
    const mid = TIMELINE_H / 2;
    const durMs = duration * 1000;

    // Hit windows: tinted bands, stronger for better scores.
    if (durMs > 0) {
      for (const m of markers) {
        const x0 = (m.timeStartMs! / durMs) * width;
        const x1 = Math.max(x0 + 2, (m.timeEndMs! / durMs) * width);
        const a = 0.15 + 0.45 * (m.score / best);
        g.fillStyle = m.modality === "video" ? `rgba(57,108,216,${a})` : `rgba(234,179,8,${a})`;
        // Audio lane on top, visual (video frames) below.
        if (m.modality === "video") g.fillRect(x0, mid, x1 - x0, mid);
        else g.fillRect(x0, 0, x1 - x0, mid);
      }
      if (active) {
        const x0 = (active.start / durMs) * width;
        const x1 = (active.end / durMs) * width;
        g.strokeStyle = "#0f0f0f";
        g.lineWidth = 2;
        g.strokeRect(x0 + 1, 1, Math.max(2, x1 - x0 - 2), TIMELINE_H - 2);
      }
    }

    // Waveform (mirrored peaks).
    if (peaks.length) {
      const top = Math.max(1e-3, ...peaks);
      g.fillStyle = "#475569";
      const step = width / peaks.length;
      for (let i = 0; i < peaks.length; i++) {
        const h = Math.max(1, (peaks[i] / top) * (mid - 4));
        g.fillRect(i * step, mid - h, Math.max(1, step - 0.5), h * 2);
      }
    } else {
      g.fillStyle = "#cbd5e1";
      g.fillRect(0, mid - 1, width, 2);
    }

    // Lane labels (only for lanes that hold matches): audio on top, visual
    // below — self-explanatory in place, so no separate legend.
    g.font = "10px -apple-system, system-ui, sans-serif";
    g.textBaseline = "top";
    const label = (text: string, y: number) => {
      const w = g.measureText(text).width + 8;
      g.fillStyle = "rgba(255,255,255,0.85)";
      g.fillRect(4, y, w, 14);
      g.fillStyle = "#475569";
      g.fillText(text, 8, y + 2);
    };
    if (markers.some((m) => m.modality === "audio")) label("Audio", 4);
    if (markers.some((m) => m.modality === "video")) label("Visual", mid + 4);

    // Playhead.
    if (duration > 0) {
      const x = (current / duration) * width;
      g.fillStyle = "#c0392b";
      g.fillRect(x - 1, 0, 2, TIMELINE_H);
    }
  });

  function seekToX(clientX: number, rect: DOMRect) {
    if (!media || !duration) return;
    const t = ((clientX - rect.left) / rect.width) * duration;
    media.currentTime = Math.min(Math.max(t, 0), duration);
    current = media.currentTime;
  }

  function onDown(e: PointerEvent) {
    const el = e.currentTarget as HTMLElement;
    const rect = el.getBoundingClientRect();
    // A click inside a hit window plays that window.
    const ms = ((e.clientX - rect.left) / rect.width) * duration * 1000;
    const hit = markers
      .filter((m) => m.timeStartMs! <= ms && ms <= m.timeEndMs!)
      .sort((a, b) => b.score - a.score)[0];
    if (hit) {
      navigate(hit.timeStartMs!, hit.timeEndMs!);
      return;
    }
    scrubbing = true;
    el.setPointerCapture(e.pointerId);
    seekToX(e.clientX, rect);
  }
  function onMove(e: PointerEvent) {
    if (scrubbing) seekToX(e.clientX, (e.currentTarget as HTMLElement).getBoundingClientRect());
  }
  function onUp() {
    scrubbing = false;
  }

  function toggle() {
    if (!media) return;
    if (media.paused) media.play();
    else media.pause();
  }

  function onKey(e: KeyboardEvent) {
    if (!media) return;
    if (e.key === " ") {
      e.preventDefault();
      toggle();
    } else if (e.key === "ArrowLeft") {
      media.currentTime = Math.max(0, media.currentTime - 5);
    } else if (e.key === "ArrowRight") {
      media.currentTime = Math.min(duration, media.currentTime + 5);
    }
  }

  /** Seek to a hit window and play it, keeping the window highlighted. */
  export function navigate(startMs: number, endMs: number) {
    active = { start: startMs, end: endMs };
    if (!media) return;
    const go = () => {
      media!.currentTime = startMs / 1000;
      media!.play().catch(() => {});
    };
    if (media.readyState >= 1) go();
    else media.addEventListener("loadedmetadata", go, { once: true });
  }
</script>

<!-- Keyboard shortcuts (space / ← / →) apply anywhere in the player. -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions, a11y_no_noninteractive_tabindex -->
<div
  class="flex flex-col h-full outline-none"
  role="application"
  aria-label={kind === "video" ? "Video player" : "Audio player"}
  tabindex="0"
  onkeydown={onKey}
>
  {#if error}
    <div class="msg">{error}</div>
  {:else if src}
    {#if kind === "video"}
      <!-- svelte-ignore a11y_media_has_caption -->
      <div class="video-wrap">
        <video
          bind:this={media}
          {src}
          controls
          playsinline
          preload="auto"
          bind:duration
          bind:currentTime={current}
          onplay={() => (playing = true)}
          onpause={() => (playing = false)}
          onerror={() => (error = "This video can't be played.")}
        ></video>
      </div>
    {:else}
      <audio
        bind:this={media}
        {src}
        preload="auto"
        bind:duration
        bind:currentTime={current}
        onplay={() => (playing = true)}
        onpause={() => (playing = false)}
        onerror={() => (error = "This audio can't be played.")}
      ></audio>
    {/if}

    <div class="controls">
      <button class="play" onclick={toggle} aria-label={playing ? "Pause" : "Play"}>
        {playing ? "❚❚" : "▶"}
      </button>
      <span class="time">{formatTime(current * 1000)} / {formatTime(duration * 1000)}</span>
      {#if active}
        <span class="window">
          Match {formatTime(active.start)}–{formatTime(active.end)}
        </span>
      {/if}
    </div>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="timeline"
      bind:clientWidth={width}
      onpointerdown={onDown}
      onpointermove={onMove}
      onpointerup={onUp}
      onpointercancel={onUp}
    >
      <canvas bind:this={canvas} style={`width:100%; height:${TIMELINE_H}px;`}></canvas>
    </div>
    {#if kind === "audio"}
      <div class="flex-1"></div>
    {/if}
  {:else}
    <div class="msg">Loading…</div>
  {/if}
</div>

<style>
  .video-wrap {
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0f0f0f;
  }
  video {
    max-width: 100%;
    max-height: 100%;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px 4px;
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .play {
    width: 32px;
    height: 28px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-bg-elevated);
  }
  .play:hover {
    background: var(--color-bg-hover);
  }
  .time {
    font-variant-numeric: tabular-nums;
  }
  .window {
    padding: 1px 6px;
    border-radius: var(--radius-sm);
    background: var(--color-score-badge);
    font-variant-numeric: tabular-nums;
  }
  .timeline {
    margin: 0 10px 10px;
    border: 1px solid var(--color-border-soft);
    border-radius: var(--radius-sm);
    background: var(--color-bg-elevated);
    cursor: pointer;
    touch-action: none;
    flex-shrink: 0;
  }
  canvas {
    display: block;
  }
  .msg {
    margin: 8px;
    font-size: 0.875rem;
    color: var(--color-text-muted);
  }
</style>
