<script lang="ts">
  import { inview, type Options } from "svelte-inview";
  import type { SearchResult } from "../types";

  interface Highlight {
    text: string;
    type: "highlight" | "normal";
  }

  const options: Options = {
    rootMargin: "50px",
  };

  export let text: string;
  export let searchResult: SearchResult;
  let highlights: Highlight[] | null = null;

  let explaining = false;

  async function explain() {
    if (explaining) {
      return;
    }
    explaining = true;
    const request = await fetch("/api/explain", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filename: searchResult.filename,
        offset: searchResult.offset,
        queries: searchResult.queries,
        preferences: searchResult.preferences,
      }),
    });
    highlights = await request.json();
  }

  let isInView = false;
  let inViewTimeout: number | null = null;
  let isInViewForEnoughTime = false;

  $: {
    if (isInView) {
      inViewTimeout = setTimeout(() => {
        isInViewForEnoughTime = true;
        explain();
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
>
  {#if highlights === null}
    {text}
  {:else}
    {#each highlights as highlight}
      {#if highlight.type === "highlight"}
        <span class="highlight">{highlight.text}</span>
      {:else}
        {highlight.text}
      {/if}
    {/each}
  {/if}
</span>

<style>
  .highlight {
    background-color: rgb(255 222 0 / 39%);
  }
</style>
