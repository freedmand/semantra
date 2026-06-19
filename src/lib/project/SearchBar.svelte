<script lang="ts">
  // Query input + relevance-feedback preference chips. Flags the query as
  // "outdated" (dashed yellow) when it has changed since the last search.
  import type { ProjectHit } from "./projectClient";

  let {
    value = $bindable(""),
    preferences,
    onSearch,
    onClearPreference,
    onSetPreference,
  }: {
    value?: string;
    preferences: Record<number, { hit: ProjectHit; weight: number }>;
    onSearch: (query: string) => void;
    onClearPreference: (hit: ProjectHit) => void;
    onSetPreference: (hit: ProjectHit, weight: number) => void;
  } = $props();

  // Active (non-zero) preferences, as chips.
  const prefList = $derived(
    Object.values(preferences).filter((p) => p.weight !== 0),
  );

  // A chip's strength is the magnitude of its signed weight; the sign carries
  // the +/- direction. Clamp to the input's 0.1–2.0 range.
  function clampMultiple(v: number): number {
    if (Number.isNaN(v)) return 1;
    return Math.min(2, Math.max(0.1, v));
  }

  // Track the (query, preferences) state at the last search to show staleness.
  let lastKey = $state("");
  const searchKey = $derived(
    JSON.stringify({ value, prefs: prefList.map((p) => [p.hit.index, p.weight]) }),
  );
  const outdated = $derived(searchKey !== lastKey);

  function doSearch() {
    onSearch(value);
    lastKey = searchKey;
  }
</script>

<div class="flex flex-1 flex-col">
  <div class="flex items-center relative flex-1">
    <input
      class="bg-white py-2 px-4 pl-12 font-mono w-full rounded-sm border"
      class:outdated
      style="color:#0f0f0f; border-color: var(--color-border);"
      placeholder="Search"
      bind:value
      onkeydown={(e) => {
        if (e.key === "Enter") doSearch();
      }}
    />
    <button class="search-button" onclick={doSearch} aria-label="Search">Search</button>
  </div>

  {#if prefList.length}
    <div class="max-h-24 overflow-y-auto mt-2 flex flex-wrap gap-2">
      {#each prefList as pref (pref.hit.index)}
        <div
          class="flex items-center font-mono rounded-sm text-sm max-w-72"
          style={pref.weight > 0 ? "background:#bfdbfe;" : "background:#fed7aa;"}
        >
          <button
            class="max-w-48 truncate px-2 py-0.5"
            title={`${pref.hit.basename}: ${pref.hit.text} (click to remove)`}
            onclick={() => onClearPreference(pref.hit)}
          >
            <span
              class="font-bold mr-1"
              style={pref.weight > 0 ? "color:#2563eb;" : "color:#f97316;"}
              >{pref.weight > 0 ? "+" : "-"}</span
            >{pref.hit.text}
          </button>
          <input
            type="number"
            class="w-12 rounded-sm border bg-white/70 px-1 py-0.5 mr-1 text-xs"
            style="color:#0f0f0f; border-color: var(--color-border);"
            min="0.1"
            max="2"
            step="0.1"
            title="Strength multiplier (0.1–2.0)"
            value={Math.abs(pref.weight)}
            onclick={(e) => e.stopPropagation()}
            onchange={(e) => {
              const v = clampMultiple((e.currentTarget as HTMLInputElement).valueAsNumber);
              onSetPreference(pref.hit, Math.sign(pref.weight) * v);
            }}
          />
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .outdated {
    background: #fefce8;
    border-style: dashed !important;
    border-color: #ca8a04 !important;
  }

  .search-button {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='25' height='24' fill='none'%3E%3Cpath stroke='%23202020' stroke-width='3' d='M10.045 13.424A7.152 7.152 0 1 0 21.003 4.23a7.152 7.152 0 0 0-10.958 9.194Zm0 0-8.984 8.984'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    text-indent: -9999px;
    width: 25px;
    height: 24px;
    position: absolute;
    left: 12px;
  }
</style>
