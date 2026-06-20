<script lang="ts">
  // Renders a hit's text. Once its per-token attribution arrives, the relevant
  // spans are tinted (reusing the embedding module's highlight presentation).
  // Quoted keyword literals from the query are additionally boxed, even before
  // the attribution loads, so exact matches always stand out.
  import {
    toSegments,
    markKeywords,
    colorFor,
    DEFAULT_HIGHLIGHT,
  } from "$lib/embedding/highlight";
  import type { Explanation } from "$lib/embedding/search";
  import { appState } from "$lib/state.svelte";

  let {
    text,
    explanation = null,
  }: { text: string; explanation?: Explanation | null } = $props();

  const segments = $derived(
    markKeywords(
      explanation
        ? toSegments(text, explanation, DEFAULT_HIGHLIGHT)
        : [{ text, weight: 0 }],
      appState.search.literals,
    ),
  );
</script>

{#each segments as seg}{#if seg.keyword}<mark
      class="rounded-sm font-semibold"
      style="background:rgb(255 224 0); box-shadow: inset 0 0 0 1px #ca8a04; color: inherit;"
      >{seg.text}</mark
    >{:else if seg.weight !== 0}<mark
      class="rounded-sm"
      style={`background:${colorFor(seg.weight)}; color: inherit;`}
      >{seg.text}</mark
    >{:else}{seg.text}{/if}{/each}
