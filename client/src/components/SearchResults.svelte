<script lang="ts">
  import type { File, SearchResult, SearchResultSet } from "../types";
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  export let searchResultSet: SearchResultSet;
  export let filesByPath: { [path: string]: File };

  function jumpToResult(file: File, searchResult: SearchResult) {
    dispatch("navigate", { file, searchResult });
  }
</script>

<div class="w-1/3 border-r-4 border-black relative">
  <div
    class="absolute left-0 top-0 right-0 bottom-0 break-words overflow-y-auto"
  >
    {#each searchResultSet as [filename, searchResults]}
      {@const file = filesByPath[filename]}
      <details open>
        <summary
          class="font-mono text-sm font-bold cursor-pointer select-none px-2 pt-2 top-0 sticky bg-slate-50"
        >
          {file.basename}
        </summary>
        <ul class="-mt-2">
          {#each searchResults as searchResult}
            <!-- svelte-ignore a11y-click-events-have-key-events -->
            <li
              on:click={() => jumpToResult(file, searchResult)}
              class="font-mono text-sm border-b last:border-0 py-4 px-4 border-black"
            >
              <span class="text-xs bg-blue-100 px-1 rounded"
                >{searchResult.distance.toFixed(2)}</span
              >
              {searchResult.text}
            </li>
          {/each}
        </ul>
      </details>
    {/each}
  </div>
</div>
