<script lang="ts">
  // Horizontal document tabs. The active tab is bound by index.
  import type { DocMeta } from "./projectClient";

  let {
    docs,
    index = $bindable(0),
    disabled = false,
  }: { docs: DocMeta[]; index?: number; disabled?: boolean } = $props();
</script>

<div
  class="flex flex-row border-b-4 relative h-10 flex-shrink-0"
  style="border-color: var(--color-border);"
>
  <div class="absolute inset-0 overflow-x-auto" style="scrollbar-width: thin;">
    <div class="inline-flex flex-nowrap flex-row items-center h-full pl-2">
      {#each docs as doc, i (doc.sha512)}
        <button
          {disabled}
          class="text-xs rounded-sm py-1 px-2 mr-2 border whitespace-nowrap"
          class:active-tab={i === index}
          style="border-color: {i === index ? 'var(--color-border)' : 'transparent'}; background: {i ===
          index
            ? 'var(--color-bg-elevated)'
            : 'transparent'};"
          onclick={() => (index = i)}
        >
          {doc.basename}
        </button>
      {/each}
    </div>
  </div>
</div>
