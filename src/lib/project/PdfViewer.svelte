<script lang="ts">
  // PDF reader: renders with PDF.js's prebuilt PDFViewer (real text layer for
  // selection), and highlights search hits with an overlay computed from
  // PDFium's per-character boxes (more accurate than PDF.js's own text layer).
  //
  // Highlight flow (`navigate`): ask the backend for rectangles in PDF
  // user-space points, scroll to the page, then place absolutely-positioned
  // boxes inside that page's element, mapping points → viewport pixels with
  // `pageView.viewport.convertToViewportRectangle(...)`. The overlay is re-applied
  // on `pagerendered` (covers lazy render + zoom re-render) so it stays aligned.
  import { onMount, onDestroy } from "svelte";
  import { convertFileSrc } from "@tauri-apps/api/core";
  import { getPdfSrc, getHighlightRects, type PageHighlight } from "./projectClient";
  import { appState } from "$lib/state.svelte";

  let { sha512 }: { sha512: string } = $props();

  let viewerContainer: HTMLDivElement;
  let pdfjsLib: any = null;
  let pdfViewer: any = null;
  let eventBus: any = null;
  let linkService: any = null;
  let pdfDoc: any = null;

  // Reactive toolbar state lives centrally; the PDF.js handles above stay local.
  const pdf = appState.search.pdf;

  // The active highlight: a 0-based page + its rects (PDF points).
  let active: { page: number; hl: PageHighlight } | null = null;
  // Set when a navigation wants the highlight scrolled into view (once).
  let needScroll = false;

  // Our highlights are painted directly onto PDF.js's page <canvas> (the only
  // way to get real blending in WKWebView — CSS `mix-blend-mode` is inert over a
  // canvas backdrop; see WebKit bug 315063). Painting is destructive, so we keep
  // a pixel snapshot of the region we covered and restore it to "erase" on any
  // change, instead of forcing PDF.js to re-rasterize the page.
  let painted:
    | { page: number; canvas: HTMLCanvasElement; clean: ImageData; x: number; y: number }
    | null = null;

  // --- Highlight appearance ------------------------------------------------
  // Painted via the canvas 2D context, so these are real compositing settings
  // (CSS `mix-blend-mode` is inert over a <canvas> in WKWebView). Color/alpha
  // match the shared page-highlight ink (`--color-page-highlight` in app.css and
  // `colorFor`'s peak in highlight.ts) so highlights look identical in the txt
  // reader, the results list, and here. `darken` over white equals normal alpha
  // for this color, and is kept so it also protects dark text/images underneath.
  const HL_COLOR = "rgb(255, 224, 0)";
  const HL_BLEND: GlobalCompositeOperation = "darken";
  const HL_ALPHA = 0.42;

  /** Scroll to and highlight a chunk's span on `page` (0-based). Safe to call
   *  before the document finishes loading — it is re-applied on pagesinit. */
  export async function navigate(page: number, pageCharStart: number, length: number) {
    try {
      const hl = await getHighlightRects(sha512, page, pageCharStart, length);
      // Erase any paint left on a page we're navigating away from.
      if (painted && painted.page !== page) restoreClean();
      active = { page, hl };
      needScroll = true;
      if (pdfViewer && pdf.totalPages > 0) {
        pdfViewer.currentPageNumber = page + 1;
        drawOverlay(); // immediate if rendered; `pagerendered` covers the rest
      }
    } catch (e) {
      console.warn("highlight failed", e);
    }
  }

  /** The rendered <canvas> for the active page, if it exists yet. */
  function activeCanvas(): HTMLCanvasElement | null {
    if (!pdfViewer || !active) return null;
    const pageView = pdfViewer.getPageView(active.page);
    const wrapper = pageView?.div?.querySelector(".canvasWrapper") as HTMLElement | null;
    return (wrapper?.querySelector("canvas") as HTMLCanvasElement | null) ?? null;
  }

  /** Undo our paint by restoring the snapshot of the clean page underneath it. */
  function restoreClean() {
    if (!painted) return;
    if (painted.canvas.isConnected) {
      try {
        painted.canvas.getContext("2d")?.putImageData(painted.clean, painted.x, painted.y);
      } catch {
        // Canvas was re-rendered/resized out from under us; it's already clean.
      }
    }
    painted = null;
  }

  /** Paint the active highlight onto the page canvas (re-painting cleanly). */
  function drawOverlay() {
    if (!active) return;
    const canvas = activeCanvas();
    if (!canvas) return; // page not rendered yet; `pagerendered` will retry
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Restore first if we already have paint on THIS canvas (e.g. re-drawing
    // after a mode/opacity tweak) so the snapshot below captures clean pixels.
    if (painted && painted.canvas === canvas) restoreClean();

    // Work in the canvas backing-store (device) pixel space: map PDFium boxes
    // (points, origin bottom-left) onto `canvas.width/height` as fractions of
    // the page. This stays aligned at any zoom because the canvas itself was
    // rendered at the current scale.
    const DW = canvas.width;
    const DH = canvas.height;
    const { pageWidth: pw, pageHeight: ph } = active.hl;
    if (!DW || !DH || !pw || !ph || active.hl.rects.length === 0) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let topT = -Infinity; // largest PDF top = visually highest point of the span
    const boxes = active.hl.rects.map(([l, b, r, t]) => {
      const x = (l / pw) * DW;
      const y = ((ph - t) / ph) * DH; // flip y: PDF up → screen down
      const w = ((r - l) / pw) * DW;
      const h = ((t - b) / ph) * DH;
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + w); maxY = Math.max(maxY, y + h);
      if (t > topT) topT = t;
      return { x, y, w, h };
    });

    // Snapshot just the band we're about to cover, so erasing is cheap.
    const sx = Math.max(0, Math.floor(minX) - 1);
    const sy = Math.max(0, Math.floor(minY) - 1);
    const bw = Math.min(DW, Math.ceil(maxX) + 1) - sx;
    const bh = Math.min(DH, Math.ceil(maxY) + 1) - sy;
    if (bw <= 0 || bh <= 0) return;

    let clean: ImageData;
    try {
      clean = ctx.getImageData(sx, sy, bw, bh);
    } catch (e) {
      console.warn("highlight snapshot failed", e);
      return;
    }

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0); // draw in raw device pixels
    ctx.globalCompositeOperation = HL_BLEND;
    ctx.globalAlpha = HL_ALPHA;
    ctx.fillStyle = HL_COLOR;
    for (const { x, y, w, h } of boxes) ctx.fillRect(x, y, w, h);
    ctx.restore();

    painted = { page: active.page, canvas, clean, x: sx, y: sy };

    if (needScroll && topT > -Infinity) {
      scrollToTop(canvas, ph, topT);
      needScroll = false;
    }
  }

  /** Scroll the viewport so the top of the highlight sits near the top. */
  function scrollToTop(canvas: HTMLCanvasElement, ph: number, topT: number) {
    const topCss = ((ph - topT) / ph) * canvas.clientHeight; // CSS px within the page
    const containerTop = viewerContainer.getBoundingClientRect().top;
    const canvasTop = canvas.getBoundingClientRect().top;
    viewerContainer.scrollTop += canvasTop - containerTop + topCss - 24; // 24px breathing room
  }

  async function loadDocument() {
    if (!pdfViewer || !sha512) return;
    pdf.loading = true;
    pdf.error = null;
    pdf.currentPage = 1;
    pdf.totalPages = 0;
    painted = null;
    active = null;
    try {
      const path = await getPdfSrc(sha512);
      const url = convertFileSrc(path);
      if (pdfDoc) {
        await pdfDoc.destroy();
        pdfDoc = null;
      }
      pdfDoc = await pdfjsLib.getDocument({ url }).promise;
      pdf.totalPages = pdfDoc.numPages;
      pdfViewer.setDocument(pdfDoc);
      linkService.setDocument(pdfDoc);
    } catch (e) {
      pdf.error = e instanceof Error ? e.message : "Failed to load PDF";
      pdf.loading = false;
    }
  }

  onMount(async () => {
    pdfjsLib = await import("pdfjs-dist");
    const viewerMod: any = await import("pdfjs-dist/web/pdf_viewer.mjs");
    await import("pdfjs-dist/web/pdf_viewer.css");

    pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
      "pdfjs-dist/build/pdf.worker.min.mjs",
      import.meta.url,
    ).href;

    eventBus = new viewerMod.EventBus();
    linkService = new viewerMod.PDFLinkService({ eventBus });
    pdfViewer = new viewerMod.PDFViewer({
      container: viewerContainer,
      eventBus,
      linkService,
      textLayerMode: 2,
    });
    linkService.setViewer(pdfViewer);

    eventBus.on("pagesinit", () => {
      pdfViewer.currentScaleValue = "page-width";
      pdf.scale = pdfViewer.currentScale;
      pdf.loading = false;
      // A navigation requested while the document was still loading: apply it now.
      if (active) {
        pdfViewer.currentPageNumber = active.page + 1;
        drawOverlay();
      }
    });
    eventBus.on("pagechanging", (e: any) => (pdf.currentPage = e.pageNumber));
    eventBus.on("scalechanging", (e: any) => (pdf.scale = e.scale));
    // Re-apply the overlay whenever the active page (re)renders (lazy load /
    // zoom). The canvas bitmap is brand-new and clean, so any prior snapshot is
    // stale — drop it and let `drawOverlay` snapshot + paint the fresh canvas.
    eventBus.on("pagerendered", (e: any) => {
      if (active && e.pageNumber === active.page + 1) {
        painted = null;
        drawOverlay();
      }
    });

    await loadDocument();
  });

  // Switching documents remounts this component (`{#key sha512}` in the parent),
  // so there's no in-place reload to handle here.

  onDestroy(() => {
    pdfViewer?.cleanup?.();
    pdfDoc?.destroy?.();
  });

  function goToPage(p: number) {
    if (pdfViewer && p >= 1 && p <= pdf.totalPages) pdfViewer.currentPageNumber = p;
  }
  function zoom(factor: number) {
    if (pdfViewer) pdfViewer.currentScale *= factor;
  }
  function fitWidth() {
    if (pdfViewer) pdfViewer.currentScaleValue = "page-width";
  }
</script>

<div class="pdf-app">
  <div class="toolbar">
    <div class="grp">
      <button onclick={() => goToPage(pdf.currentPage - 1)} disabled={pdf.currentPage <= 1}>‹</button>
      <input
        type="number"
        min="1"
        max={pdf.totalPages}
        value={pdf.currentPage}
        onchange={(e) => goToPage(parseInt((e.target as HTMLInputElement).value) || 1)}
      />
      <span class="dim">/ {pdf.totalPages}</span>
      <button onclick={() => goToPage(pdf.currentPage + 1)} disabled={pdf.currentPage >= pdf.totalPages}
        >›</button
      >
    </div>
    <div class="grp">
      <button onclick={() => zoom(0.8)} title="Zoom out">−</button>
      <span class="dim">{Math.round(pdf.scale * 100)}%</span>
      <button onclick={() => zoom(1.25)} title="Zoom in">+</button>
      <button onclick={fitWidth} title="Fit width" aria-label="Fit width">
        <svg viewBox="0 0 83 83" width="15" height="15" fill="none" stroke="currentColor" aria-hidden="true">
          <path d="M40.4026 42.3164C44.2692 46.5388 49.8275 49.1866 56.0042 49.1866C67.6847 49.1866 77.1536 39.7177 77.1536 28.0372C77.1536 16.3566 67.6847 6.8877 56.0042 6.8877C44.3236 6.8877 34.8547 16.3566 34.8547 28.0372C34.8547 33.541 36.9571 38.5538 40.4026 42.3164ZM40.4026 42.3164L21.6421 61.0768" stroke-width="6.33594" stroke-linecap="round" />
          <path d="M37.7332 58.4917L55.8199 76.5783M42.9757 76.5783H55.8199V63.7342" stroke-width="5.69523" />
          <path d="M26.3554 45.647L8.26876 27.5603M21.1129 27.5603H8.26876V40.4045" stroke-width="5.69523" />
        </svg>
      </button>
    </div>
  </div>

  <div class="viewer-area">
    {#if pdf.loading}<div class="overlay">Loading PDF…</div>{/if}
    {#if pdf.error}<div class="overlay error">{pdf.error}</div>{/if}
    <div bind:this={viewerContainer} class="viewer-container">
      <div class="pdfViewer"></div>
    </div>
  </div>
</div>

<style>
  .pdf-app {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    overflow: hidden;
  }
  .toolbar {
    display: flex;
    gap: 1.5rem;
    align-items: center;
    padding: 6px 12px;
    border-bottom: 1px solid var(--color-border-soft);
    background: var(--color-bg-elevated);
    flex-shrink: 0;
  }
  .grp {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .toolbar button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 26px;
    height: 26px;
    border: 1px solid var(--color-border-soft);
    border-radius: var(--radius-sm);
    background: var(--color-bg-elevated);
    color: var(--color-text);
    cursor: pointer;
  }
  .toolbar button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .toolbar input {
    width: 44px;
    height: 26px;
    text-align: center;
    font-size: 0.85rem;
    border: 1px solid var(--color-border-soft);
    border-radius: var(--radius-sm);
    background: var(--color-bg);
    color: var(--color-text);
  }
  .dim {
    color: var(--color-text-muted);
    font-size: 0.85rem;
  }
  .viewer-area {
    flex: 1;
    position: relative;
    min-height: 0;
    overflow: hidden;
  }
  .viewer-container {
    position: absolute;
    inset: 0;
    overflow: auto;
    background: #525659;
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.55);
    color: white;
    z-index: 50;
  }
  .overlay.error {
    color: #fca5a5;
  }

  /* Highlights are painted directly onto the page <canvas> (see drawOverlay) —
     no DOM overlay elements, so there are no highlight styles here. */
</style>
