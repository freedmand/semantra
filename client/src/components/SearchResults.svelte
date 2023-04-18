<script lang="ts">
  import {
    preferenceKey,
    type File,
    type Preference,
    type SearchResult,
    type SearchResultSet,
  } from "../types";
  import { createEventDispatcher } from "svelte";
  import SearchResultText from "./SearchResultText.svelte";
  const dispatch = createEventDispatcher();

  export let searchResultSet: SearchResultSet;
  export let filesByPath: { [path: string]: File };
  export let preferences: { [key: string]: Preference };

  function jumpToResult(file: File, searchResult: SearchResult) {
    dispatch("navigate", { file, searchResult });
  }

  function setPreference(
    e: Event,
    file: File,
    searchResult: SearchResult,
    weight: number
  ) {
    dispatch("setPreference", {
      file,
      searchResult,
      weight,
    });
    e.stopPropagation();
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
            {@const prefKey = preferenceKey(file, searchResult)}
            {@const preference = preferences[prefKey] || {
              file,
              searchResult,
              weight: 0,
            }}
            {#key searchResult}
              <!-- svelte-ignore a11y-click-events-have-key-events -->
              <li
                on:click={() => jumpToResult(file, searchResult)}
                class="font-mono text-sm border-b last:border-0 py-4 px-4 border-black pointer"
              >
                <span class="text-xs highlight px-1 rounded"
                  >{searchResult.distance.toFixed(2)}</span
                >
                <button
                  class="bg-gray-300 rounded px-2"
                  class:bg-blue-200={preference.weight > 0}
                  class:text-blue-600={preference.weight > 0}
                  class:font-bold={preference.weight > 0}
                  on:click={(e) =>
                    setPreference(
                      e,
                      file,
                      searchResult,
                      preference.weight > 0 ? 0 : 1
                    )}>+</button
                >
                <button
                  class="bg-gray-300 rounded px-2"
                  class:bg-orange-200={preference.weight < 0}
                  class:text-orange-500={preference.weight < 0}
                  class:font-bold={preference.weight < 0}
                  on:click={(e) =>
                    setPreference(
                      e,
                      file,
                      searchResult,
                      preference.weight < 0 ? 0 : -1
                    )}>-</button
                >
                <SearchResultText {searchResult} text={searchResult.text} />
              </li>
            {/key}
          {/each}
        </ul>
      </details>
    {/each}
  </div>
</div>

<style>
  .highlight {
    background: rgb(255 222 0 / 39%);
  }
</style>
