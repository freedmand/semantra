<script lang="ts">
  // Renders one search result's text, optionally highlighted by which words
  // contributed most to the semantic similarity. Until an `explanation` arrives
  // (it's fetched in a batch after the search resolves) it shows plain text, so
  // results appear instantly and light up a moment later. All display logic
  // lives in `highlight.ts`; this component just renders the segments.
  import type { Explanation } from "./search";
  import {
    toSegments,
    colorFor,
    DEFAULT_HIGHLIGHT,
    type HighlightOptions,
  } from "./highlight";

  let {
    text,
    explanation = null,
    options = DEFAULT_HIGHLIGHT,
  }: {
    text: string;
    explanation?: Explanation | null;
    options?: HighlightOptions;
  } = $props();

  const segments = $derived(
    explanation ? toSegments(text, explanation, options) : null,
  );
</script>

<span class="text"
  >{#if segments}{#each segments as seg}{#if seg.weight}<mark
          style:background-color={colorFor(seg.weight)}
          title={`contribution ${seg.score >= 0 ? "+" : ""}${seg.score.toFixed(4)}`}
          >{seg.text}</mark
        >{:else}{seg.text}{/if}{/each}{:else}{text}{/if}</span
>

<style>
  .text {
    flex: 1 1 auto;
    font-size: 0.92em;
    line-height: 1.5;
  }

  mark {
    color: inherit;
    background-color: transparent;
    border-radius: 3px;
    padding: 0.05em 0.1em;
  }
</style>
