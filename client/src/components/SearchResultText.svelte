<script lang="ts">
  import { onMount } from "svelte";
  import type { SearchResult } from "../types";

  interface Highlight {
    text: string;
    type: "highlight" | "normal";
  }

  export let text: string;
  export let searchResult: SearchResult;
  let highlights: Highlight[] | null = null;

  onMount(async () => {
    setTimeout(async () => {
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
      console.log(highlights);
    }, 0);
  });
</script>

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

<style>
  .highlight {
    background-color: rgb(255 222 0 / 39%);
  }
</style>
