<script lang="ts">
  // Horizontal document tabs. Reads the document list and the active index from
  // the central search state.
  import { appState } from "$lib/state.svelte";

  const search = appState.search;
</script>

<div
  class="flex flex-row border-b-4 relative h-10 flex-shrink-0"
  style="border-color: var(--color-border);"
>
  <div class="absolute inset-0 overflow-x-auto" style="scrollbar-width: thin;">
    <div class="inline-flex flex-nowrap flex-row items-center h-full pl-2">
      {#each search.docs as doc, i (doc.sha512)}
        <button
          class="text-xs rounded-sm py-1 px-2 mr-2 border whitespace-nowrap"
          class:active-tab={i === search.activeIndex}
          style="border-color: {i === search.activeIndex ? 'var(--color-border)' : 'transparent'}; background: {i ===
          search.activeIndex
            ? 'var(--color-bg-elevated)'
            : 'transparent'};"
          onclick={() => (search.activeIndex = i)}
        >
          {doc.basename}
        </button>
      {/each}
    </div>
  </div>
</div>
