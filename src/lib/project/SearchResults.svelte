<script lang="ts">
  // Results sidebar. A top rail (same height as the document tab bar) holds the
  // view tabs — "Show by files" (collapsible groups) vs "Show by results" (flat
  // excerpts) — plus an expand/collapse-all control (file view) and a filter
  // toggle. The toggle reveals a panel with the filename filter and a "filter to
  // the current file" button. Grouping/excerpt are derived subviews on appState.
  import { appState, loadMoreResults, scorePercent } from "$lib/state.svelte";
  import SearchResult from "./SearchResult.svelte";

  const search = appState.search;

  // Whether the filter panel is exposed. Opens automatically once a filter is
  // active so the active filter stays visible after a reload.
  let filterOpen = $state(false);
  const filterActive = $derived(search.filenameFilter !== "" || search.filterViewed);

  let loadingMore = $state(false);
  async function more() {
    if (loadingMore) return;
    loadingMore = true;
    try {
      await loadMoreResults();
    } finally {
      loadingMore = false;
    }
  }

  function matchesFilter(basename: string): boolean {
    return basename.toLowerCase().includes(search.filenameFilter.toLowerCase());
  }

  // Filtered subviews + the count of what's actually visible, so we can show a
  // dedicated empty state when a search (or its filter) yields nothing.
  const filteredGroups = $derived(
    appState.searchGroups.filter((g) => matchesFilter(g.basename)),
  );
  const filteredExcerpt = $derived(
    appState.searchExcerpt.filter((h) => matchesFilter(h.basename)),
  );
  const visibleCount = $derived(
    search.excerptView ? filteredExcerpt.length : filteredGroups.length,
  );
</script>

<div
  class="w-1/3 max-lg:w-72 flex flex-col items-stretch flex-shrink-0 border-r-4"
  style="background: var(--color-bg); border-color: var(--color-border);"
>
  <!-- Rail: view tabs + filter toggle, matching the document tab bar's height. -->
  <div class="flex items-center gap-2 h-10 px-2 flex-shrink-0">
    <div class="segmented">
      <button
        class="seg"
        class:active={!search.excerptView}
        onclick={() => (search.excerptView = false)}
      >
        Show by files
      </button>
      <button
        class="seg"
        class:active={search.excerptView}
        onclick={() => (search.excerptView = true)}
      >
        Show by results
      </button>
    </div>
    <div class="flex-1"></div>
    {#if !search.excerptView && visibleCount > 0}
      <button
        class="rail-btn"
        title="Expand / collapse all"
        onclick={() => {
          search.detailReverse = !search.detailReverse;
          search.closed = {};
        }}
      >
        {search.detailReverse ? "Expand all" : "Collapse all"}
      </button>
    {/if}
    <button
      class="icon-btn filter-toggle-icon"
      class:active={filterOpen || filterActive}
      title="Filter files"
      onclick={() => (filterOpen = !filterOpen)}
      aria-label="Filter files"
    ></button>
  </div>

  <!-- Filter panel (revealed by the rail's filter toggle). -->
  {#if filterOpen}
    <div
      class="flex items-center gap-2 px-2 py-2 border-b flex-shrink-0"
      style="border-color: var(--color-border-soft);"
    >
      <div class="flex-1 flex items-center relative">
        <input
          class="border rounded-sm bg-white py-1 pl-8 pr-2 font-mono w-full text-sm"
          style="border-color: var(--color-border); color: #0f0f0f;"
          placeholder="Filter files"
          bind:value={search.filenameFilter}
        />
        <div class="icon filter-icon" aria-hidden="true"></div>
      </div>
      <button
        class="filter-to-doc-btn"
        class:active={search.filterViewed}
        disabled={appState.searchActiveDoc == null}
        onclick={() => (search.filterViewed = !search.filterViewed)}
      >
        {search.filterViewed ? "Show all files" : "Filter to current file"}
      </button>
    </div>
  {/if}

  <!-- List -->
  <div class="flex-1 relative">
    <div class="absolute inset-0 break-words overflow-y-auto pb-2">
      {#if search.unsearched}
        <div class="m-2 font-mono text-sm" style="color: var(--color-text-muted);">
          Enter a search query above and press Enter.
        </div>
      {:else if visibleCount === 0}
        <div class="m-2 font-mono text-sm" style="color: var(--color-text-muted);">
          {#if appState.searchIndexing}
            No results yet — documents are still indexing…
          {:else}
            No search results
          {/if}
        </div>
      {:else if search.excerptView}
        <ul>
          {#each filteredExcerpt as hit (hit.index)}
            <SearchResult
              {hit}
              explanation={search.explanations[hit.index] ?? null}
              showFilename={true}
            />
          {/each}
        </ul>
      {:else}
        {#each filteredGroups as group (group.sha512)}
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
                >{scorePercent(group.score)}</span
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
        {/each}
      {/if}

      {#if appState.searchHasMore && visibleCount > 0}
        <div class="px-2 pt-2">
          <button class="load-more-btn" onclick={more} disabled={loadingMore}>
            {loadingMore ? "Loading…" : "Load more results"}
          </button>
        </div>
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

  /* View switch ("Show by files" / "Show by results") as a segmented control:
     one outline around both, a divider between, the active half filled. */
  .segmented {
    display: inline-flex;
    flex-shrink: 0;
    height: 28px;
    box-sizing: border-box;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }
  .seg {
    display: inline-flex;
    align-items: center;
    font-size: 0.8rem;
    line-height: 1;
    padding: 0 0.65rem;
    background: transparent;
    color: var(--color-text-muted);
    white-space: nowrap;
    cursor: pointer;
  }
  .seg + .seg {
    border-left: 1px solid var(--color-border);
  }
  .seg:hover:not(.active) {
    background: var(--color-bg-hover);
    color: var(--color-text);
  }
  .seg.active {
    background: var(--color-bg-elevated);
    color: var(--color-text);
  }

  /* "Expand / collapse all" button in the rail (file view), same height as the
     segmented control and filter button. */
  .rail-btn {
    display: inline-flex;
    align-items: center;
    flex-shrink: 0;
    height: 28px;
    box-sizing: border-box;
    font-size: 0.8rem;
    line-height: 1;
    padding: 0 0.6rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: white;
    color: var(--color-text);
    cursor: pointer;
  }
  .rail-btn:hover {
    background: var(--color-bg-hover);
  }

  /* "Filter to current file" text button inside the filter panel. */
  .filter-to-doc-btn {
    flex-shrink: 0;
    font-size: 0.8rem;
    line-height: 1;
    padding: 0.4rem 0.6rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: white;
    color: var(--color-text);
    white-space: nowrap;
    cursor: pointer;
  }
  .filter-to-doc-btn:hover:not(:disabled) {
    background: var(--color-bg-hover);
  }
  .filter-to-doc-btn.active {
    background: var(--color-bg-hover);
    font-weight: 600;
  }
  .filter-to-doc-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }

  /* "Load more results" button at the foot of the list. */
  .load-more-btn {
    width: 100%;
    font-size: 0.85rem;
    padding: 0.45rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: white;
    color: var(--color-text);
    cursor: pointer;
  }
  .load-more-btn:hover:not(:disabled) {
    background: var(--color-bg-hover);
  }
  .load-more-btn:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .icon-btn {
    width: 34px;
    height: 28px;
    flex-shrink: 0;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background-color: white;
    background-position: center;
    background-repeat: no-repeat;
    background-size: 60%;
    cursor: pointer;
  }
  .icon-btn:hover {
    background-color: var(--color-bg-hover);
  }
  .icon-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .icon-btn.active {
    background-color: #e2e8f0;
  }

  /* Same funnel mark as the input's inline icon, centered as a button glyph. */
  .filter-toggle-icon {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOSIgaGVpZ2h0PSIxNSIgZmlsbD0ibm9uZSI+PHBhdGggc3Ryb2tlPSIjMDAwIiBzdHJva2Utd2lkdGg9IjIiIGQ9Ik04LjYyNSAxMy44MzNoMS4yMk0uMjgyIDEuNTQyaDE3LjkwN002Ljk1NiA5LjczNmg0LjU1OE0zLjY2IDUuNjM5aDExLjE1Ii8+PC9zdmc+");
  }
</style>
