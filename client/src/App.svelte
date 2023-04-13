<script lang="ts">
  import { onMount, tick } from "svelte";
  import SearchResults from "./components/SearchResults.svelte";
  import TextView from "./components/TextView.svelte";
  import Tailwindcss from "./Tailwind.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import type {
    File,
    Navigation,
    PdfPosition,
    SearchResult,
    SearchResultSet,
  } from "./types";
  import PdfView from "./components/PdfView.svelte";
  import TabBar from "./components/TabBar.svelte";

  let files: File[] = [];
  let activeFileIndex = 0;
  let tokens: string[] = [];
  let text: string | null = null;
  let pdfPositions: PdfPosition[] = [];
  let updating = false;

  $: activeFile =
    activeFileIndex < files.length ? files[activeFileIndex] : null;
  $: filesByPath = Object.fromEntries(
    files.map((file) => [file.filename, file])
  );
  $: fileIndicesByPath = Object.fromEntries(
    files.map((file, index) => [file.filename, index])
  );

  $: updateFile(activeFile);

  async function updateFile(file: File | null) {
    // Reset everything
    tokens = [];
    text = null;
    pdfPositions = [];

    if (file == null) return;

    updating = true;

    // Get text
    const textResponse = await fetch(
      `/api/text?filename=${encodeURIComponent(file.filename)}`
    );
    tokens = await textResponse.json();
    text = tokens.join("");

    if (file.filetype === "pdf") {
      const pdfResponse = await fetch(
        `/api/pdfpositions?filename=${encodeURIComponent(file.filename)}`
      );
      pdfPositions = await pdfResponse.json();
    }

    await tick();
    navigate();

    updating = false;
  }

  let searchResultSet: SearchResultSet = [];

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
    searchResultSet = await response.json();
  }

  onMount(async () => {
    const filesResponse = await fetch("/api/files");
    files = await filesResponse.json();
  });

  $: tokenOffsets = tokens.reduce(
    (acc, token) => {
      const lastOffset = acc[acc.length - 1];
      acc.push(lastOffset + token.length);
      return acc;
    },
    [0]
  );

  let pendingNavigation: Navigation | null = null;

  function navigate() {
    if (pendingNavigation == null) return;
    if (textView) {
      textView.navigate(
        tokenOffsets[pendingNavigation.searchResult.offset[0]],
        tokenOffsets[pendingNavigation.searchResult.offset[1]]
      );
    } else if (pdfView) {
      pdfView.navigate(
        tokenOffsets[pendingNavigation.searchResult.offset[0]],
        tokenOffsets[pendingNavigation.searchResult.offset[1]]
      );
    }
    pendingNavigation = null;
  }

  async function jumpToResult(result: Navigation) {
    pendingNavigation = result;
    const newFileIndex = fileIndicesByPath[result.file.filename];
    if (newFileIndex !== activeFileIndex) {
      activeFileIndex = newFileIndex;
    } else {
      navigate();
    }
  }
</script>

<Tailwindcss />

<main class="flex flex-col h-full bg-slate-100">
  <header class="flex flex-row items-center border-b-4 border-black py-4 px-8">
    <h1 class="text-3xl font-mono font-bold inline-flex pr-6">Semantra</h1>
    <SearchBar on:search={(e) => handleSearch(e.detail)} />
  </header>
  <article class="flex flex-1 flex-row relative items-stretch">
    <SearchResults
      on:navigate={(e) => jumpToResult(e.detail)}
      {filesByPath}
      {searchResultSet}
    />
    <div class="flex flex-col w-2/3">
      {#if activeFile != null}
        <TabBar disabled={updating} bind:index={activeFileIndex} {files} />
        {#if activeFile.filetype === "text"}
          <TextView
            bind:this={textView}
            text={text == null ? "Loading..." : text}
          />
        {:else if activeFile.filetype === "pdf"}
          <PdfView
            bind:this={pdfView}
            file={activeFile}
            positions={pdfPositions}
          />
        {/if}
      {:else}
        <div class="text-gray-600 ml-2 mt-2 text-sm">Loading...</div>
      {/if}
    </div>
  </article>
</main>

<style>
  :global(html, body) {
    @apply h-full;
  }
</style>
