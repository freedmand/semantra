<script lang="ts">
  export let pageNumber: number;
  export let scales: number[];

  let scaleIndex = 0;

  function getSrc(scale: number) {
    return `/api/pdfpage?page=${pageNumber}&scale=${scale}`;
  }

  function handleLoad() {
    if (scaleIndex !== scales.length - 1) {
      scaleIndex++;
    }
  }
</script>

<img
  on:load={handleLoad}
  draggable="false"
  class="absolute left-0 top-0 right-0 bottom-0 w-full h-full object-contain select-none pointer-events-none"
  src={getSrc(scales[scaleIndex])}
  alt="Page {pageNumber + 1}"
/>
