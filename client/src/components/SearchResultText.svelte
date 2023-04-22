<script lang="ts">
  import { inview, type Options } from "svelte-inview";
  import type { Highlight, SearchResult } from "../types";
  import { explainDictionary, requestExplanation } from "../explainQueue";

  let element: HTMLSpanElement;

  const options: Options = {
    rootMargin: "50px",
  };

  export let text: string;
  export let searchResult: SearchResult;
  let highlights: Highlight[] | null = null;

  $: params = {
    filename: searchResult.filename,
    offset: searchResult.offset,
    queries: searchResult.queries,
    preferences: searchResult.preferences,
    text: text,
  };

  $: highlights = $explainDictionary[JSON.stringify(params)];

  let isInView = false;
  let inViewTimeout: number | null = null;
  let isInViewForEnoughTime = false;

  $: {
    if (isInView) {
      inViewTimeout = setTimeout(() => {
        isInViewForEnoughTime = true;
        requestExplanation(element, params);
      }, 100);
    } else {
      isInViewForEnoughTime = false;
      if (inViewTimeout !== null) {
        clearTimeout(inViewTimeout);
        inViewTimeout = null;
      }
    }
  }
</script>

<span
  use:inview={options}
  on:inview_change={(e) => (isInView = e.detail.inView)}
  bind:this={element}
>
  {#if highlights == null}
    <span>{text}</span>
  {:else}
    {#each highlights as highlight}
      {#if highlight.type === "highlight"}
        <span class="explain-highlight">{highlight.text}</span>
      {:else}
        <span>{highlight.text}</span>
      {/if}
    {/each}
  {/if}
</span>

<style>
  .explain-highlight {
    background-color: rgb(154 134 0 / 18%);
  }
</style>
