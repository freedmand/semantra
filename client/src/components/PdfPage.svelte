<script lang="ts">
  import { inview, type Options } from "svelte-inview";
  import type { File, Offset, PdfPosition } from "../types";
  import PdfPageImage from "./PdfPageImage.svelte";
  import PdfChars from "./PdfChars.svelte";

  const options: Options = {
    rootMargin: "50px",
  };

  export let file: File;
  export let position: PdfPosition;
  export let pageNumber: number;
  export let selectedOffset: Offset | null;
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
  on:inview_change={(e) => (isInView = e.detail.inView)}
  class="bg-white my-4 mx-auto bg-contain relative"
  style="width: {position.page_width}px; height: {position.page_height}px;"
>
  {#if isInView && isInViewForEnoughTime}
    <PdfPageImage {file} {pageNumber} scales={[0.25, 2]} />
    <PdfChars {file} {position} {pageNumber} {selectedOffset} />
  {/if}
</div>
