<script lang="ts">
  import { onMount, tick } from "svelte";
  import SearchResults from "./components/SearchResults.svelte";
  import TextView from "./components/TextView.svelte";
  import Tailwindcss from "./Tailwind.svelte";
  import SearchBar from "./components/SearchBar.svelte";
  import {
    type File,
    type Navigation,
    type PdfPosition,
    type SearchResultSet,
    type Preference,
    preferenceKey,
    type ParsedQuery,
  } from "./types";
  import PdfView from "./components/PdfView.svelte";
  import TabBar from "./components/TabBar.svelte";

  let files: File[] = [];
  let activeFileIndex = 0;
  let tokens: string[] = [];
  let text: string | null = null;
  let pdfPositions: PdfPosition[] = [];
  let updating = false;
  let unsearched = true;
  let searchResultsElem: SearchResults;

  let preferences: { [key: string]: Preference } = {};

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

  let searchResultSet: SearchResultSet = {
    results: [],
    sort: "asc",
  };

  let textView: TextView;
  let pdfView: PdfView;
  let searchBar: SearchBar;

  export function parseQuery(query: string): ParsedQuery[] {
    // Parse the query
    // e.g. "dog + cat" => [{query: "dog", weight: 1}, {query: "cat", weight: 1}]
    // e.g. "dog - cat" => [{query: "dog", weight: 1}, {query: "cat", weight: -1}]
    // e.g. "dog is cool - cat" => [{query: "dog is cool", weight: 1}, {query: "cat", weight: -1}]
    // e.g. "dog +1.2 cat" => [{query: "dog", weight: 1}, {query: "cat", weight: 1.2}]
    // e.g. "+3 dogs are nice -2 cats are mean" => [{query: "dogs are nice", weight: 3}, {query: "cats are mean", weight: 2}]
    // Parse the query
    const regex = /([\+\-]?\d*\.?\d*\s*)?([^\+\-]+)/g;
    const parsedQueries: ParsedQuery[] = [];

    let match;
    while ((match = regex.exec(query)) !== null) {
      const weight =
        parseFloat(match[1]) || (match[1] && match[1].includes("-") ? -1 : 1);
      const searchTerm = match[2].trim();
      parsedQueries.push({ query: searchTerm, weight });
    }

    return parsedQueries;
  }

  function scrollSearchResultsToTop() {
    if (searchResultsElem) searchResultsElem.scrollToTop();
  }

  async function handleSearch(query: string) {
    const preferenceValues = Object.values(preferences)
      .filter((preference) => preference.weight !== 0)
      .map((x) => ({ ...x }));

    // Ignore empty queries
    if (query.trim() === "" && preferenceValues.length === 0) {
      searchResultSet = [];
      scrollSearchResultsToTop();
      return;
    }
    const parsedQueries = parseQuery(query);

    // Adjust weights so that all positive weights are split evenly
    // and all negative weights are split evenly, and the sum of all
    // weights is 1
    const POSITIVE_RATIO = 0.61803398875;
    const NEGATIVE_RATIO = 1 - POSITIVE_RATIO;

    const totalPositiveCount =
      parsedQueries.filter((query) => query.weight > 0).length +
      preferenceValues.filter((preference) => preference.weight > 0).length;
    const totalNegativeCount =
      parsedQueries.filter((query) => query.weight < 0).length +
      preferenceValues.filter((preference) => preference.weight < 0).length;
    for (const query of parsedQueries) {
      if (query.weight > 0) {
        query.weight *= POSITIVE_RATIO / totalPositiveCount;
      } else if (query.weight < 0) {
        query.weight *= NEGATIVE_RATIO / totalNegativeCount;
      }
    }
    for (const preference of preferenceValues) {
      if (preference.weight > 0) {
        preference.weight *= POSITIVE_RATIO / totalPositiveCount;
      } else if (preference.weight < 0) {
        preference.weight *= NEGATIVE_RATIO / totalNegativeCount;
      }
    }

    const response = await fetch("/api/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        queries: parsedQueries,
        preferences: preferenceValues,
      }),
    });
    searchResultSet = await response.json();
    sidebarExpanded = true;
    scrollSearchResultsToTop();
    unsearched = false;
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

  async function navigate() {
    if (pendingNavigation == null) return;
    sidebarExpanded = false;
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

    await tick();
    // Scroll active tab into view
    const activeTab = document.querySelector(".active-tab");
    if (activeTab) {
      activeTab.scrollIntoView({
        inline: "center",
      });
    }
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

  function setPreference(preference: Preference) {
    preferences[preferenceKey(preference.file, preference.searchResult)] =
      preference;
    if (searchBar != null) searchBar.scrollToBottomOfPreferences();
  }

  let sidebarExpanded = true;
</script>

<Tailwindcss />

<main class="flex flex-col h-full bg-slate-100">
  <header
    class="flex flex-row border-b-4 border-black py-4 px-8 max-lg:px-4 items-start"
  >
    <h1 class="text-3xl font-mono font-bold inline-flex pr-6 mt-1">Semantra</h1>
    <SearchBar
      bind:this={searchBar}
      {preferences}
      on:setPreference={(e) => setPreference(e.detail)}
      on:search={(e) => handleSearch(e.detail)}
    />
  </header>
  <article class="flex flex-1 flex-row relative items-stretch">
    <SearchResults
      bind:sidebarExpanded
      bind:this={searchResultsElem}
      {unsearched}
      {preferences}
      on:setPreference={(e) => setPreference(e.detail)}
      on:navigate={(e) => jumpToResult(e.detail)}
      {activeFile}
      {filesByPath}
      {searchResultSet}
    />
    <div class="flex flex-col flex-1">
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
  <footer class="bg-black text-white py-1 px-4 text-sm">
    <a
      class="underline mr-4"
      href="https://github.com/freedmand/semantra/blob/main/docs/help.md"
      target="_blank">Help</a
    >
    <a
      class="underline mr-4"
      href="https://github.com/freedmand/semantra/blob/main/docs/tutorial.md"
      target="_blank">Tutorial</a
    >
    <a
      class="underline"
      href="https://github.com/freedmand/semantra"
      target="_blank">Source code</a
    >
  </footer>
</main>

<style>
  :global(html, body) {
    @apply h-full;
  }
</style>
