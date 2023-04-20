<script lang="ts">
  import type {
    File,
    Preference,
    SearchResult,
    SearchResultSet,
    ScoredSearchResult,
  } from "../types";
  import SearchResultComponent from "./SearchResult.svelte";

  export let searchResultSet: SearchResultSet;
  export let filesByPath: { [path: string]: File };
  export let preferences: { [key: string]: Preference };
  export let activeFile: File | null;

  let filterViewed = false;
  let excerptView = false;

  let searchContainer: HTMLDivElement;

  function scrollToTop() {
    if (searchContainer) searchContainer.scrollTop = 0;
  }

  function noop(..._args: any[]) {
    // Don't do anything
    // This can be used to mark reactive deps
  }

  $: {
    // Scroll to top when results change
    noop(searchResultSet, filterViewed, excerptView);
    scrollToTop();
  }

  function getScore(searchResults: SearchResult[]): number {
    let total = 0;
    for (const searchResult of searchResults) {
      total += searchResult.distance;
    }
    return total / searchResults.length;
  }

  $: scoredSearchResultSet = searchResultSet
    .map<ScoredSearchResult>(([filename, searchResults]) => [
      filename,
      searchResults,
      getScore(searchResults),
    ])
    .sort((a, b) => a[2] - b[2])
    .filter(([filename]) => {
      if (filterViewed && activeFile != null) {
        return filename === activeFile.filename;
      } else {
        return true;
      }
    });
  $: sortedSearchResults = searchResultSet
    .map((x) => x[1])
    .flat()
    .sort((a, b) => a.distance - b.distance)
    .filter((searchResult) => {
      if (filterViewed && activeFile != null) {
        return searchResult.filename === activeFile.filename;
      } else {
        return true;
      }
    });
</script>

<div class="w-1/3 border-r-4 border-black flex flex-col items-stretch">
  <div>
    <button
      class="bg-gray-300 rounded px-2 m-2"
      disabled={activeFile == null}
      on:click={() => (filterViewed = !filterViewed)}
    >
      {#if filterViewed}
        Show all files
      {:else}Filter to only viewed file
      {/if}</button
    >
    <button
      class="bg-gray-300 rounded px-2 m-2"
      on:click={() => (excerptView = !excerptView)}
    >
      {#if excerptView}
        Show file view
      {:else}Show exercept view{/if}</button
    >
  </div>
  <div class="flex-1 relative">
    <div
      class="absolute left-0 top-0 right-0 bottom-0 break-words overflow-y-auto"
      bind:this={searchContainer}
    >
      {#if excerptView}
        <!-- Excerpt view -->
        <ul class="-mt-2">
          {#each sortedSearchResults as searchResult}
            {@const file = filesByPath[searchResult.filename]}
            {#key searchResult}
              <SearchResultComponent
                on:navigate
                on:setPreference
                {file}
                {searchResult}
                {preferences}
                showFilename={true}
              />
            {/key}
          {/each}
        </ul>
      {:else}
        <!-- File view -->
        {#each scoredSearchResultSet as [filename, searchResults, score]}
          {@const file = filesByPath[filename]}
          {#key [filename, searchResults, score]}
            <details open>
              <summary
                class="font-mono text-sm font-bold cursor-pointer select-none px-2 pt-2 top-0 sticky bg-slate-50"
              >
                {file.basename}
                <span class="text-xs highlight px-1 rounded"
                  >{score.toFixed(2)}</span
                >
              </summary>
              <ul class="-mt-2">
                {#each searchResults as searchResult}
                  {#key searchResult}
                    <SearchResultComponent
                      on:navigate
                      on:setPreference
                      {file}
                      {searchResult}
                      {preferences}
                      showFilename={false}
                    />
                  {/key}
                {/each}
              </ul>
            </details>
          {/key}
        {/each}
      {/if}
    </div>
  </div>
</div>

<style>
  .highlight {
    background: rgb(255 222 0 / 39%);
  }
</style>
