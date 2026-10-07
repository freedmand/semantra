<script lang="ts">
  // Horizontal document tabs. Reads the document list and the active index from
  // the central search state.
  import { appState } from "$lib/state.svelte";

  const search = appState.search;

  /** A small type glyph so mixed media projects scan at a glance. */
  const glyph: Record<string, string> = {
    pdf: "▤",
    text: "¶",
    csv: "▦",
    image: "▣",
    audio: "♪",
    video: "▶",
  };
</script>

<div
  class="flex flex-row border-b-4 relative h-10 flex-shrink-0"
  style="border-color: var(--color-border);"
>
  <div class="absolute inset-0 overflow-x-auto" style="scrollbar-width: thin;">
    <div class="inline-flex flex-nowrap flex-row items-center h-full pl-2">
      {#each search.docs as doc, i (doc.sha512)}
        <button
          class="text-xs rounded-sm h-7 inline-flex items-center px-2 mr-2 border whitespace-nowrap"
          class:active-tab={i === search.activeIndex}
          style="border-color: {i === search.activeIndex ? 'var(--color-border)' : 'transparent'}; background: {i ===
          search.activeIndex
            ? 'var(--color-bg-elevated)'
            : 'transparent'};"
          onclick={() => (search.activeIndex = i)}
        >
          <span class="mr-1" style="color: var(--color-text-dim);" aria-hidden="true"
            >{glyph[doc.filetype] ?? ""}</span
          >{doc.basename}
        </button>
      {/each}
    </div>
  </div>
</div>
