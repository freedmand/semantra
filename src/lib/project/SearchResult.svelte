<script lang="ts">
  // One search result row: score badge, +/- relevance-feedback buttons, and the
  // (optionally highlighted) snippet. Clicking the row navigates into the doc.
  // Per-row data (`hit`, `explanation`) comes in as props; preference state and
  // the navigate/set-preference actions come from the central appState.
  import type { Explanation } from "$lib/embedding/search";
  import type { ProjectHit } from "./projectClient";
  import { appState, setPreference, jumpToResult } from "$lib/state.svelte";
  import SearchResultText from "./SearchResultText.svelte";

  let {
    hit,
    explanation = null,
    showFilename = false,
  }: {
    hit: ProjectHit;
    explanation?: Explanation | null;
    showFilename?: boolean;
  } = $props();

  const preference = $derived(appState.search.preferences[hit.index]?.weight ?? 0);

  function setPref(e: Event, weight: number) {
    e.stopPropagation();
    setPreference(hit, weight);
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<li
  class="font-mono text-sm border-b last:border-0 py-4 px-4 cursor-pointer"
  style="border-color: var(--color-border);"
  onclick={() => jumpToResult(hit)}
>
  <div
    class="border-l-4 pl-2 -ml-2"
    style={`border-color: ${preference > 0 ? "#2563eb" : preference < 0 ? "#f97316" : "transparent"};`}
  >
    {#if showFilename}
      <div class="font-bold text-base my-2">
        {hit.basename}
        <span class="px-1 rounded-sm text-xs" style="background: var(--color-score-badge);"
          >{hit.score.toFixed(2)}</span
        >
      </div>
    {:else}
      <span class="text-xs px-1 rounded-sm" style="background: var(--color-score-badge);"
        >{hit.score.toFixed(2)}</span
      >
    {/if}
    <button
      class="rounded-sm px-2"
      style={preference > 0
        ? "background:#bfdbfe; color:#2563eb; font-weight:700;"
        : "background:#d1d5db;"}
      onclick={(e) => setPref(e, preference > 0 ? 0 : 1)}>+</button
    >
    <button
      class="rounded-sm px-2"
      style={preference < 0
        ? "background:#fed7aa; color:#f97316; font-weight:700;"
        : "background:#d1d5db;"}
      onclick={(e) => setPref(e, preference < 0 ? 0 : -1)}>-</button
    >
    <SearchResultText text={hit.text} {explanation} />
  </div>
</li>
