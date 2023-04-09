<script lang="ts">
  import type { SearchResult } from "../types";
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  export let searchResults: SearchResult[];

  function jumpToResult(searchResult: SearchResult) {
    dispatch("navigate", searchResult);
  }
</script>

<div class="w-1/3 border-r-4 border-black relative">
  <ul
    class="absolute left-0 top-0 right-0 bottom-0 break-words overflow-y-auto"
  >
    {#each searchResults as searchResult}
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <li
        on:click={() => jumpToResult(searchResult)}
        class="font-mono text-sm border-b last:border-0 py-4 px-4 border-black"
      >
        <span class="text-xs bg-blue-100 px-1 rounded"
          >{searchResult.distance.toFixed(2)}</span
        >
        {searchResult.text}
      </li>
    {/each}
  </ul>
</div>
