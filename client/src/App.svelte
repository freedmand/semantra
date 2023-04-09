<script lang="ts">
  import { onMount } from "svelte";
  import SearchResults from "./components/SearchResults.svelte";
  import TextView from "./components/TextView.svelte";
  import Tailwindcss from "./Tailwind.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import type { SearchResult } from "./types";

  let tokens: string[] = [];
  let text: string | null = null;

  let searchResults: SearchResult[] = [];

  let textView: TextView;

  async function handleSearch(query: string) {
    // Ignore empty queries
    if (query.trim() === "") return;

    const response = await fetch("/api/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });
    searchResults = await response.json();
  }

  onMount(async () => {
    const response = await fetch("/api/text");
    tokens = await response.json();
    text = tokens.join("");
  });

  $: tokenOffsets = tokens.reduce(
    (acc, token, index) => {
      const lastOffset = acc[acc.length - 1];
      acc.push(lastOffset + token.length);
      return acc;
    },
    [0]
  );

  function jumpToResult(searchResult: SearchResult) {
    textView.navigate(
      tokenOffsets[searchResult.offset[0]],
      tokenOffsets[searchResult.offset[1]]
    );
  }
</script>

<Tailwindcss />

<main class="flex flex-col h-full">
  <header class="flex flex-row items-center border-b-4 border-black py-4 px-8">
    <h1 class="text-3xl font-mono font-bold inline-flex pr-6">Semantra</h1>
    <SearchBar on:search={(e) => handleSearch(e.detail)} />
  </header>
  <article class="flex flex-1 flex-row relative items-stretch">
    <SearchResults
      on:navigate={(e) => jumpToResult(e.detail)}
      {searchResults}
    />
    <TextView bind:this={textView} text={text == null ? "Loading..." : text} />
  </article>
</main>

<style>
  :global(html, body) {
    @apply h-full;
  }
</style>
