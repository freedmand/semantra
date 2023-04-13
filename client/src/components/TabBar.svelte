<script lang="ts">
  import type { File } from "../types";

  export let files: File[];
  export let index: number;
  export let disabled: boolean;
</script>

<div
  class="flex flex-row border-b-4 border-black py-1 pl-2 overflow-x-auto scroll thin-scroll"
>
  {#each files as file, i}
    <button
      {disabled}
      class:active={i === index}
      class="text-xs rounded py-1 px-2 mr-2 border border-transparent"
      on:click={() => (index = i)}
    >
      {file.basename}
    </button>
  {/each}
</div>

<style>
  .thin-scroll {
    scrollbar-width: thin;
  }

  .active {
    @apply bg-white border-black;
  }
</style>
