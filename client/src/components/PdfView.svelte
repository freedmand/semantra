<script lang="ts">
  import { tick } from "svelte";
  import type { File, Offset, PdfPosition } from "../types";
  import PdfPage from "./PdfPage.svelte";

  export let positions: PdfPosition[];
  export let file: File;

  let currentPage = 1;
  $: numPages = positions.length;

  let pageContainer: HTMLDivElement;

  let zoom = 1;
  const MAX_ZOOM = 4;
  const MIN_ZOOM = 0.4;

  let selectedOffset: Offset | null = null;

  let pagesInView: { [page: number]: boolean } = {};

  $: sortedPages = Object.entries(pagesInView)
    .filter(([_, isInView]) => isInView)
    .sort(([a, _], [b, __]) => parseInt(a) - parseInt(b));
  $: pageObjects = sortedPages.map<[number, Element]>(([page, _]) => [
    parseInt(page),
    pageContainer.children[parseInt(page)],
  ]);

  $: pageObjects.length > 0 && updateCurrentPage();

  async function adjustZoom(newZoom: number) {
    if (newZoom > MAX_ZOOM || newZoom < MIN_ZOOM) {
      return;
    }
    const oldZoom = zoom;
    let scrollTop = pageContainer.scrollTop;
    zoom = newZoom;
    await tick();
    pageContainer.scrollTop = scrollTop * (newZoom / oldZoom);
  }

  function getPageNumber(offset: number): [number, number] {
    let page = 0;
    for (; page < positions.length; page++) {
      const positionOffset = positions[page].char_index;
      if (offset < positionOffset) {
        return [page - 1, offset - positions[page - 1].char_index];
      }
    }
    return [page - 1, offset - positions[page - 1].char_index];
  }

  // When true, highlights will scroll into view
  let scrollHighlights = false;

  export function navigate(start: number, end: number) {
    selectedOffset = [start, end];

    // Jump to selection
    const [pageNumber, _] = getPageNumber(start);
    pageContainer.children[pageNumber].scrollIntoView();
    scrollHighlights = true;
  }

  function updatePageNumber({
    isInView,
    pageNumber,
  }: {
    isInView: boolean;
    pageNumber: number;
  }) {
    pagesInView[pageNumber] = isInView;
    pagesInView = { ...pagesInView };
  }

  function updateCurrentPage() {
    const windowHeight = window.innerHeight;
    for (const [pageNumber, page] of pageObjects) {
      if (page == null) {
        continue;
      }
      const rect = page.getBoundingClientRect();
      if (rect.bottom >= windowHeight / 2) {
        currentPage = pageNumber + 1;
        return;
      }
    }
    if (pageObjects.length > 0) {
      currentPage = pageObjects[0][0] + 1;
    }
  }

  function jumpToCurrentPage() {
    const elem = pageContainer.children[currentPage - 1];
    if (elem != null) {
      elem.scrollIntoView();
    }
  }
</script>

<div class="relative flex-1">
  <div class="absolute left-0 top-0 right-0 bottom-0">
    <div
      bind:this={pageContainer}
      class="absolute left-0 top-0 right-0 bottom-0 bg-gray-400 w-full h-full overflow-auto"
      on:scroll={updateCurrentPage}
    >
      {#each positions as position, pageNumber}
        <PdfPage
          {file}
          {pageNumber}
          {position}
          {selectedOffset}
          {zoom}
          bind:scrollHighlights
          on:inview={(e) => updatePageNumber(e.detail)}
        />
      {/each}
    </div>
    {#if numPages > 0}
      <div class="absolute left-2 bottom-2">
        <div class="text-left">
          <div
            class="bg-slate-100 border-2 border-black text-center font-mono font-bold text-2xl inline-block"
          >
            <button
              class="w-8 h-8 align-middle"
              on:click={() => adjustZoom(zoom + 0.1)}>+</button
            ><br />
            <button
              class="w-8 h-8 align-middle"
              on:click={() => adjustZoom(zoom - 0.1)}>-</button
            >
          </div>
        </div>
        <div
          class="bg-slate-100 border-2 border-black font-mono font-bold text-sm mt-2 p-2"
        >
          <input
            class="border border-gray-300 mr-1 pl-1"
            style="width: 7ch;"
            type="number"
            size="4"
            min="1"
            max={numPages}
            bind:value={currentPage}
            on:input={jumpToCurrentPage}
          />/{numPages}
        </div>
      </div>
    {/if}
  </div>
</div>
