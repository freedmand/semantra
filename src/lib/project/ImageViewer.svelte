<script lang="ts">
  // Image reader: the file itself, fit to the pane, with wheel/pinch zoom
  // (anchored at the cursor), drag-to-pan, and Fit / 100% controls. Formats the
  // webview can't decode (camera RAW, JPEG 2000) fall back to a large
  // backend-rendered preview. `navigate()` pulses the frame so a search hit on
  // the image reads as "this one".
  import { getFileUrl, getThumbnail } from "./projectClient";

  let { sha512 }: { sha512: string } = $props();

  let src = $state<string | null>(null);
  let usedPreview = false;
  let error = $state<string | null>(null);
  let natural = $state({ w: 0, h: 0 });
  let box = $state({ w: 0, h: 0 });
  let scale = $state(1);
  let offset = $state({ x: 0, y: 0 });
  let fitted = $state(true);
  let pulse = $state(false);
  let drag: { x: number; y: number; ox: number; oy: number } | null = null;

  $effect(() => {
    let cancelled = false;
    getFileUrl(sha512)
      .then((u) => !cancelled && (src = u))
      .catch((e) => !cancelled && (error = `${e}`));
    return () => (cancelled = true);
  });

  async function onError() {
    if (usedPreview) {
      error = "This image format can't be displayed.";
      return;
    }
    usedPreview = true;
    try {
      src = await getThumbnail(sha512, { maxSide: 2048 });
    } catch (e) {
      error = `${e}`;
    }
  }

  function fitScale() {
    if (!natural.w || !box.w) return 1;
    return Math.min(box.w / natural.w, box.h / natural.h, 1);
  }

  function fit() {
    fitted = true;
    scale = fitScale();
    offset = { x: (box.w - natural.w * scale) / 2, y: (box.h - natural.h * scale) / 2 };
  }

  function zoomAt(factor: number, cx: number, cy: number) {
    const next = Math.min(Math.max(scale * factor, fitScale() * 0.5), 8);
    const k = next / scale;
    offset = { x: cx - (cx - offset.x) * k, y: cy - (cy - offset.y) * k };
    scale = next;
    fitted = false;
  }

  // Keep the image fitted as the pane resizes, until the user zooms.
  $effect(() => {
    void box.w;
    void box.h;
    void natural.w;
    if (fitted) fit();
  });

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    // Trackpad pinch arrives as ctrl+wheel with small deltas.
    const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002));
    zoomAt(factor, e.clientX - r.left, e.clientY - r.top);
  }

  function onDown(e: PointerEvent) {
    drag = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }
  function onMove(e: PointerEvent) {
    if (!drag) return;
    offset = { x: drag.ox + e.clientX - drag.x, y: drag.oy + e.clientY - drag.y };
    fitted = false;
  }
  function onUp() {
    drag = null;
  }

  /** A search hit landed on this image: re-fit and pulse the frame. */
  export function navigate() {
    fit();
    pulse = false;
    requestAnimationFrame(() => (pulse = true));
    setTimeout(() => (pulse = false), 1200);
  }
</script>

<div class="flex flex-col h-full">
  <div class="toolbar">
    <button class="tb-btn" onclick={() => zoomAt(1 / 1.25, box.w / 2, box.h / 2)} aria-label="Zoom out">−</button>
    <span class="zoom">{Math.round(scale * 100)}%</span>
    <button class="tb-btn" onclick={() => zoomAt(1.25, box.w / 2, box.h / 2)} aria-label="Zoom in">+</button>
    <button class="tb-btn" onclick={fit}>Fit</button>
    <button
      class="tb-btn"
      onclick={() => {
        zoomAt(1 / scale, box.w / 2, box.h / 2);
      }}>100%</button
    >
    {#if natural.w}
      <span class="dims">{natural.w} × {natural.h}</span>
    {/if}
  </div>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="stage"
    bind:clientWidth={box.w}
    bind:clientHeight={box.h}
    onwheel={onWheel}
    onpointerdown={onDown}
    onpointermove={onMove}
    onpointerup={onUp}
    onpointercancel={onUp}
    ondblclick={fit}
  >
    {#if error}
      <div class="msg">{error}</div>
    {:else if src}
      <img
        {src}
        alt=""
        draggable="false"
        class:pulse
        onload={(e) => {
          const img = e.currentTarget as HTMLImageElement;
          natural = { w: img.naturalWidth, h: img.naturalHeight };
          fit();
        }}
        onerror={onError}
        style={`width:${natural.w * scale}px; height:${natural.h * scale}px; transform: translate(${offset.x}px, ${offset.y}px);`}
      />
    {:else}
      <div class="msg">Loading…</div>
    {/if}
  </div>
</div>

<style>
  .toolbar {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 36px;
    padding: 0 8px;
    border-bottom: 1px solid var(--color-border-soft);
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .tb-btn {
    min-width: 28px;
    height: 24px;
    padding: 0 6px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-bg-elevated);
  }
  .tb-btn:hover {
    background: var(--color-bg-hover);
  }
  .zoom {
    min-width: 3.5em;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }
  .dims {
    margin-left: auto;
    color: var(--color-text-muted);
    font-variant-numeric: tabular-nums;
  }
  .stage {
    position: relative;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    cursor: grab;
    background: repeating-conic-gradient(#e2e8f0 0% 25%, #f8fafc 0% 50%) 50% / 20px 20px;
    touch-action: none;
  }
  .stage:active {
    cursor: grabbing;
  }
  img {
    position: absolute;
    left: 0;
    top: 0;
    max-width: none;
    transform-origin: 0 0;
    user-select: none;
    box-shadow: 0 1px 6px rgb(0 0 0 / 25%);
  }
  img.pulse {
    animation: pulse 1.2s ease-out;
  }
  @keyframes pulse {
    0% {
      outline: 6px solid var(--color-page-highlight);
    }
    100% {
      outline: 6px solid transparent;
    }
  }
  .msg {
    margin: 8px;
    font-size: 0.875rem;
    color: var(--color-text-muted);
  }
</style>
