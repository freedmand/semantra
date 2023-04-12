<script lang="ts">
  import { onMount } from "svelte";
  import SearchResults from "./components/SearchResults.svelte";
  import TextView from "./components/TextView.svelte";
  import Tailwindcss from "./Tailwind.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import type { PdfPosition, SearchResult } from "./types";
  import PdfView from "./components/PdfView.svelte";

  let tokens: string[] = [];
  let text: string | null = null;
  let filetype: "text" | "pdf" | null;
  let pdfPositions: PdfPosition[] = [];

  let searchResults: SearchResult[] = [];

  let textView: TextView;
  let pdfView: PdfView;

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
    const filetypeResponse = await fetch("/api/filetype");
    filetype = await filetypeResponse.json();

    if (filetype === "pdf") {
      const response = await fetch("/api/pdfpositions");
      pdfPositions = await response.json();
    }

    const response = await fetch("/api/text");
    tokens = await response.json();
    text = tokens.join("");
  });

  $: tokenOffsets = tokens.reduce(
    (acc, token) => {
      const lastOffset = acc[acc.length - 1];
      acc.push(lastOffset + token.length);
      return acc;
    },
    [0]
  );

  function jumpToResult(searchResult: SearchResult) {
    if (textView) {
      textView.navigate(
        tokenOffsets[searchResult.offset[0]],
        tokenOffsets[searchResult.offset[1]]
      );
    } else if (pdfView) {
      pdfView.navigate(
        tokenOffsets[searchResult.offset[0]],
        tokenOffsets[searchResult.offset[1]]
      );
    }
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
    {#if filetype == "text"}
      <TextView
        bind:this={textView}
        text={text == null ? "Loading..." : text}
      />
    {:else if filetype === "pdf"}
      <PdfView bind:this={pdfView} positions={pdfPositions} />
    {/if}
  </article>
</main>

<style>
  :global(html, body) {
    @apply h-full;
  }
</style>
