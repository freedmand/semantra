<script lang="ts">
  // CSV reader: renders the parsed grid with canvas-datagrid (a single-canvas
  // grid that virtualizes to large row counts). Search hits navigate by
  // (row, col) — see `ProjectHit.page`/`pageCharStart`, repurposed for CSV — so
  // `navigate` scrolls the matched cell into view and paints it with the shared
  // page-highlight ink, matching the PDF/text readers.
  //
  // Like PdfViewer, the library is dynamically imported in onMount (it touches
  // the DOM at load) and the component is remounted per document via `{#key}` in
  // the parent, so there's no in-place reload to handle.
  import { onMount, onDestroy } from "svelte";
  import { getCsvData, type CsvData } from "./projectClient";

  let { sha512 }: { sha512: string } = $props();

  let container: HTMLDivElement;
  let grid: any = null;
  let loadError = $state<string | null>(null);

  // The cell to highlight, set by `navigate`. Plain (non-reactive) vars: the grid
  // repaints imperatively via grid.draw(), and the highlight handler closes over
  // them, so Svelte reactivity isn't involved. -1 means "no highlight".
  let hlRow = -1;
  let hlCol = -1;
  // The grid's own 2D context (grid.canvas.getContext), captured once so the
  // afterrendercell handler can paint the target cell's highlight.
  let gridCtx: CanvasRenderingContext2D | null = null;

  // Translucent highlight ink, matching the PDF reader (--color-page-highlight /
  // PdfViewer's HL_COLOR @ HL_ALPHA). Painted over the cell's real background and
  // under its text, so the matched cell reads like a highlighter pass.
  const HL_FILL = "rgba(255, 224, 0, 0.42)";

  /** Resolve a Semantra CSS custom property to its current value. canvas-datagrid
   *  paints on a <canvas> and can't read CSS vars itself, so we hand it values. */
  const cssVar = (name: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /** Scroll to and highlight the cell at (row, col). Safe to call before the grid
   *  finishes building — the target is stored and applied once it's ready. */
  export function navigate(row: number, col: number) {
    hlRow = row;
    hlCol = col;
    applyHighlight();
  }

  function applyHighlight() {
    if (!grid || hlRow < 0) return;
    try {
      grid.scrollIntoView(hlCol, hlRow); // (x = column, y = row)
    } catch {
      // Out-of-range (e.g. a stale hit) — leave the view where it is.
    }
    grid.draw();
  }

  /** Recolor the grid (drawn on canvas, so styled via JS) to Semantra's theme:
   *  elevated cells with soft gridlines, page-bg headers for contrast, dim row
   *  numbers, accent-tinted selection. Values come from the app's CSS vars. */
  function themeGrid() {
    const s = grid.style;
    const text = cssVar("--color-text");
    const muted = cssVar("--color-text-muted");
    const bg = cssVar("--color-bg");
    const elevated = cssVar("--color-bg-elevated");
    const hover = cssVar("--color-bg-hover");
    const border = cssVar("--color-border-soft");
    const accentMuted = cssVar("--color-accent-muted");

    s.cellColor = text;
    s.cellBackgroundColor = elevated;
    s.cellBorderColor = border;
    s.cellHoverColor = text;
    s.cellHoverBackgroundColor = hover;
    s.cellSelectedColor = text;
    s.cellSelectedBackgroundColor = accentMuted;

    s.columnHeaderCellColor = text;
    s.columnHeaderCellBackgroundColor = bg;
    s.columnHeaderCellBorderColor = border;
    s.columnHeaderCellHoverColor = text;
    s.columnHeaderCellHoverBackgroundColor = hover;

    s.rowHeaderCellColor = muted;
    s.rowHeaderCellBackgroundColor = bg;
    s.rowHeaderCellBorderColor = border;
    s.cornerCellBackgroundColor = bg;
    s.cornerCellBorderColor = border;
    s.gridBorderColor = border;

    const font = "13px ui-sans-serif, system-ui, -apple-system, sans-serif";
    s.cellFont = font;
    s.columnHeaderCellFont = font;
    s.rowHeaderCellFont = font;
  }

  async function build() {
    let data: CsvData;
    try {
      data = await getCsvData(sha512);
    } catch (e) {
      loadError = e instanceof Error ? e.message : "Failed to load CSV";
      return;
    }

    const mod: any = await import("canvas-datagrid");
    const canvasDatagrid = mod.default ?? mod;

    // Unique internal column keys (`name`) with the original header as the display
    // `title`, so blank or duplicate headers don't collide as data-object keys.
    const schema = data.headers.map((h, i) => ({ name: `c${i}`, title: h }));
    // Rows as objects keyed by the internal column name; cell values stay verbatim.
    const rows = data.rows.map((r) => {
      const o: Record<string, string> = {};
      for (let i = 0; i < schema.length; i++) o[`c${i}`] = r[i] ?? "";
      return o;
    });

    grid = canvasDatagrid({
      parentNode: container,
      data: rows,
      schema,
      // Keep (row, col) stable so navigation lands on the right cell: no reorder,
      // no sort (sorting would desync rowIndex from the backend's row numbering).
      allowColumnReordering: false,
      allowSorting: false,
    });
    grid.style.height = "100%";
    grid.style.width = "100%";
    themeGrid();

    // canvas-datagrid draws into this single <canvas>; we paint the highlight
    // through its context. (Do NOT assign to e.cell.style — it's a style *name*
    // string, and writing to it throws inside the render loop and aborts the
    // draw, dropping the header/row-header until the next scroll.)
    gridCtx = grid.canvas?.getContext?.("2d") ?? null;

    // Overlay the translucent ink on the matched cell after its background is
    // drawn but before its text, so the cell keeps its theme background and the
    // text stays crisp on top. The hlRow/hlCol >= 0 guard avoids matching the
    // header and corner cells, whose rowIndex/columnIndex are -1 — the same
    // sentinel as "nothing highlighted" (which previously tinted the corner).
    grid.addEventListener("afterrendercell", (e: any) => {
      if (!gridCtx || hlRow < 0 || hlCol < 0) return;
      if (e.cell.rowIndex !== hlRow || e.cell.columnIndex !== hlCol) return;
      gridCtx.save();
      gridCtx.fillStyle = HL_FILL;
      gridCtx.fillRect(e.cell.x, e.cell.y, e.cell.width, e.cell.height);
      gridCtx.restore();
    });

    // A navigation requested while the grid was still loading: apply it now.
    applyHighlight();
  }

  onMount(build);
  onDestroy(() => {
    try {
      grid?.dispose?.();
    } catch {
      // Best-effort teardown; the node is being removed anyway.
    }
  });
</script>

<div class="csv-app">
  {#if loadError}
    <div class="overlay error">{loadError}</div>
  {/if}
  <div bind:this={container} class="grid-container"></div>
</div>

<style>
  .csv-app {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    overflow: hidden;
    position: relative;
  }
  .grid-container {
    flex: 1;
    min-height: 0;
    width: 100%;
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.55);
    color: #fca5a5;
    z-index: 50;
  }
</style>
