<script lang="ts">
  import type { File } from "../types";

  export let files: File[];
  export let index: number;
  export let disabled: boolean;
</script>

<div class="flex flex-row border-b-4 border-black relative h-10">
  <div
    class="absolute left-0 top-0 right-0 bottom-0 overflow-x-auto thin-scroll"
  >
    <div class="inline-flex flex-nowrap flex-row items-center h-full pl-2">
      {#each files as file, i}
        <button
          {disabled}
          class:active-tab={i === index}
          class="text-xs rounded py-1 px-2 mr-2 border border-transparent"
          on:click={() => (index = i)}
        >
          {file.basename}
        </button>
      {/each}
    </div>
  </div>
</div>

<style>
  .thin-scroll {
    scrollbar-width: thin;
  }

  .active-tab {
    @apply bg-white border-black;
  }
</style>
