<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { inview, type Options } from "svelte-inview";
  import type { File, Offset, PdfPosition } from "../types";
  import PdfPageImage from "./PdfPageImage.svelte";
  import PdfChars from "./PdfChars.svelte";
  const dispatch = createEventDispatcher();

  const options: Options = {
    rootMargin: "50px",
  };

  export let file: File;
  export let position: PdfPosition;
  export let pageNumber: number;
  export let selectedOffset: Offset | null;
  export let zoom: number;
  export let scrollHighlights: boolean;
  const marginPx = 16;
  let isInView = false;
  let isInViewForEnoughTime = false;

  $: {
    if (isInView) {
      setTimeout(() => {
        isInViewForEnoughTime = true;
      }, 100);
    } else {
      isInViewForEnoughTime = false;
    }
  }
</script>

<div
  use:inview={options}
  on:inview_change={(e) => {
    isInView = e.detail.inView;
    dispatch("inview", {
      isInView,
      pageNumber,
    });
  }}
  class="bg-white bg-contain relative"
  style="width: {position.page_width * zoom}px; height: {position.page_height *
    zoom}px; margin: {marginPx * zoom}px auto {marginPx * zoom}px auto"
>
  {#if isInView && isInViewForEnoughTime}
    <PdfPageImage {file} {pageNumber} scales={[0.25, 2]} />
    <PdfChars
      {file}
      {position}
      {pageNumber}
      {selectedOffset}
      {zoom}
      bind:scrollHighlights
    />
  {/if}
</div>
