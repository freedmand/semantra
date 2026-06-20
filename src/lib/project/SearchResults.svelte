<script lang="ts">
  // Results sidebar: file-grouped (collapsible) or flat excerpt view, with a
  // filename filter, expand/collapse-all, and a "solo the active doc" toggle.
  // Reads the central search state directly; grouping/excerpt are derived
  // subviews on appState. Ported from semantra-web's SearchResults.
  import { appState } from "$lib/state.svelte";
  import SearchResult from "./SearchResult.svelte";

  const search = appState.search;

  function matchesFilter(basename: string): boolean {
    return basename.toLowerCase().includes(search.filenameFilter.toLowerCase());
  }
</script>

<div
  class="w-1/3 max-lg:w-72 flex flex-col items-stretch flex-shrink-0 border-r-4"
  style="background: var(--color-bg); border-color: var(--color-border);"
>
  <!-- Controls -->
  <div class="flex items-center gap-1 mb-2 pr-2 mt-2 pl-2">
    <div class="flex-1 flex items-center relative">
      <input
        class="border rounded-sm bg-white py-1 pl-8 font-mono flex-1 w-full"
        style="border-color: var(--color-border); color: #0f0f0f;"
        placeholder="Filter files"
        bind:value={search.filenameFilter}
      />
      <div class="icon filter-icon" aria-hidden="true"></div>
    </div>
    {#if !search.excerptView}
      <button
        class="icon-btn toggle-detail-icon"
        title="Expand / collapse all"
        onclick={() => {
          search.detailReverse = !search.detailReverse;
          search.closed = {};
        }}
        aria-label="Expand or collapse all"
      ></button>
    {/if}
    <button
      class="icon-btn solo-icon"
      class:active={search.filterViewed}
      disabled={appState.searchActiveDoc == null}
      title={search.filterViewed ? "Show all files" : "Filter to current document"}
      onclick={() => (search.filterViewed = !search.filterViewed)}
      aria-label="Filter to current document"
    ></button>
    <button
      class="icon-btn toggle-view-icon"
      title="Toggle file / excerpt view"
      onclick={() => (search.excerptView = !search.excerptView)}
      aria-label="Toggle view"
    ></button>
  </div>

  <!-- List -->
  <div class="flex-1 relative">
    <div class="absolute inset-0 break-words overflow-y-auto pb-2">
      {#if search.unsearched}
        <div class="m-2 font-mono text-sm" style="color: var(--color-text-muted);">
          Enter a search query above and press Enter.
        </div>
      {:else if search.excerptView}
        <ul>
          {#each appState.searchExcerpt as hit (hit.index)}
            {#if matchesFilter(hit.basename)}
              <SearchResult
                {hit}
                explanation={search.explanations[hit.index] ?? null}
                showFilename={true}
              />
            {/if}
          {/each}
        </ul>
      {:else}
        {#each appState.searchGroups as group (group.sha512)}
          {#if matchesFilter(group.basename)}
            <details open={search.detailReverse ? search.closed[group.sha512] : !search.closed[group.sha512]}
              ontoggle={(e) => {
                const open = (e.target as HTMLDetailsElement).open;
                search.closed = {
                  ...search.closed,
                  [group.sha512]: search.detailReverse ? open : !open,
                };
              }}
            >
              <summary
                class="font-mono font-bold cursor-pointer select-none px-2 pt-2 sticky top-0"
                style="background: var(--color-bg);"
              >
                {group.basename}
                <span class="text-xs px-1 rounded-sm" style="background: var(--color-score-badge);"
                  >{group.score.toFixed(2)}</span
                >
              </summary>
              <ul>
                {#each group.hits as hit (hit.index)}
                  <SearchResult
                    {hit}
                    explanation={search.explanations[hit.index] ?? null}
                    showFilename={false}
                  />
                {/each}
              </ul>
            </details>
          {/if}
        {/each}
      {/if}
    </div>
  </div>
</div>

<style>
  /* Filename-filter input icon. */
  .filter-icon {
    position: absolute;
    left: 8px;
    top: 7px;
    width: 18px;
    height: 15px;
    background-repeat: no-repeat;
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOSIgaGVpZ2h0PSIxNSIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9IjIiIGQ9Ik04LjYyNSAxMy44MzNoMS4yMk0uMjgyIDEuNTQyaDE3LjkwN002Ljk1NiA5LjczNmg0LjU1OE0zLjY2IDUuNjM5aDExLjE1Ii8+PC9zdmc+");
  }

  .icon-btn {
    width: 34px;
    height: 30px;
    flex-shrink: 0;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background-color: white;
    background-position: center;
    background-repeat: no-repeat;
    background-size: 65%;
    cursor: pointer;
  }
  .icon-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .icon-btn.active {
    background-color: #e2e8f0;
  }

  .solo-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIxMyIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBkPSJNMTAuMTMzIDEuNDhjLTMuNTg1IDAtNC44MzQgMS40NzMtOC45MDggNS4yMDcgNC4wNzQgMy43MzQgNS4zMjMgNS4yMDggOC45MDggNS4yMDggMy41ODUgMCA0LjgzNC0xLjQ3NCA4LjkwOC01LjIwOC00LjA3NC0zLjczNC01LjMyMy01LjIwNy04LjkwOC01LjIwN1oiLz48Y2lyY2xlIGN4PSIxMC4xMzMiIGN5PSI2LjY4NyIgcj0iMi4zMDUiIGZpbGw9IiMwMDAiLz48Y2lyY2xlIGN4PSIxMC4xMzMiIGN5PSI2LjY4NyIgcj0iMy42OTQiIHN0cm9rZT0iIzAwMCIvPjwvc3ZnPg==");
  }
  .toggle-view-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIxNiIgZmlsbD0ibm9uZSI+PHBhdGggZmlsbD0iIzAwMCIgZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNLjIgMWgxMHYxSC4zVjFabTEwLjMgOC45aDguNlYxMWgtOC42VjkuOVptLS4yLTdIMi44VjRoNy41VjNabS4yIDguOWg4LjZ2MWgtOC42di0xWm0tLjItN0gyLjhWNmg3LjVWNVptLjIgOC45aDguNnYxaC04LjZ2LTFabS0uMi03SDIuOHYxLjFoNy41di0xWk0xNCA0LjZsLjkgMS0xIC45LTItMi4zLS4zLS41LjUtLjRMMTQgMS42bC44IDEtLjguN0E1IDUgMCAwIDEgMTcgNC42YzEgMSAxLjMgMi4zIDEuMyAzLjhoLTEuM2MwLTEuMy0uMy0yLjItMS0yLjktLjQtLjQtMS0uOC0xLjgtMVptLTggNy4yIDEgMWMtLjktLjItMS41LS41LTEuOS0xLS42LS42LTEtMS42LTEtMi44SDNjMCAxLjQuMyAyLjggMS4zIDMuOEE1IDUgMCAwIDAgNyAxNGwtLjguNy45IDFMOSAxNGwuNC0uNC0uNC0uNS0yLTIuMy0xIC45WiIgY2xpcC1ydWxlPSJldmVub2RkIi8+PC9zdmc+");
  }
  .toggle-detail-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOSIgaGVpZ2h0PSIxNSIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9IjIiIGQ9Ik01LjIuNHY5LjdNLjQgNS4zSDEwTTkuMiAxMy43aDguOSIvPjwvc3ZnPg==");
  }
</style>
