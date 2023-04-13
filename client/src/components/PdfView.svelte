<script lang="ts">
  import type { File, Offset, PdfPosition } from "../types";
  import PdfPage from "./PdfPage.svelte";

  export let positions: PdfPosition[];
  export let file: File;

  let pageContainer: HTMLDivElement;

  let selectedOffset: Offset | null = null;

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

  export function navigate(start: number, end: number) {
    selectedOffset = [start, end];

    // Jump to selection
    const [pageNumber, _] = getPageNumber(start);
    pageContainer.children[pageNumber].scrollIntoView();
  }
</script>

<div class="relative flex-1">
  <div
    bind:this={pageContainer}
    class="absolute left-0 top-0 right-0 bottom-0 bg-gray-500 w-full h-full overflow-auto"
  >
    {#each positions as position, pageNumber}
      <PdfPage {file} {pageNumber} {position} {selectedOffset} />
    {/each}
  </div>
</div>
