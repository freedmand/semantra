<script lang="ts">
  // The launched project workspace: header + search bar, file-grouped results
  // sidebar with relevance feedback, document tabs, and a reader (ProseMirror
  // for text, PDF.js for PDFs) that highlights the matched passage. Reachable at
  // `/search?id=<uuid>`. All state lives in appState.search; live indexing updates
  // (new tabs as documents finish) arrive via the root layout's subscription.
  //
  // The only thing kept local is the imperative navigation glue: the `bind:this`
  // viewer refs and the retry loop that drives them. It's registered on
  // `appState.search.requestNavigate` so result rows can trigger it centrally.
  import { onMount, tick } from "svelte";
  import { goto } from "$app/navigation";
  import { getDocumentText, type ProjectHit } from "$lib/project/projectClient";
  import { appState, loadSearch, runSearch } from "$lib/state.svelte";
  import SearchBar from "$lib/project/SearchBar.svelte";
  import SearchResults from "$lib/project/SearchResults.svelte";
  import TabBar from "$lib/project/TabBar.svelte";
  import TextView from "$lib/project/TextView.svelte";
  import PdfViewer from "$lib/project/PdfViewer.svelte";
  import CsvViewer from "$lib/project/CsvViewer.svelte";
  import IndexProgressBar from "$lib/project/IndexProgressBar.svelte";
  import SemantraLogo from "$lib/SemantraLogo.svelte";

  const search = appState.search;

  let textViewRef = $state<any>(null);
  let pdfViewerRef = $state<any>(null);
  let csvViewerRef = $state<any>(null);

  function tryNavigate() {
    const hit = search.pendingNav;
    const activeDoc = appState.searchActiveDoc;
    if (!hit || !activeDoc || activeDoc.sha512 !== hit.sha512) return;
    if (activeDoc.filetype === "pdf") {
      if (!pdfViewerRef) return;
      pdfViewerRef.navigate(hit.page ?? 0, hit.pageCharStart, hit.charEnd - hit.charStart);
      search.pendingNav = null;
    } else if (activeDoc.filetype === "csv") {
      if (!csvViewerRef) return;
      // CSV reuses page/pageCharStart as (row, col).
      csvViewerRef.navigate(hit.page ?? 0, hit.pageCharStart);
      search.pendingNav = null;
    } else {
      // `textViewRef` only binds once the `{#await}` resolves and `TextView`
      // mounts, so its presence already means the text is loaded.
      if (!textViewRef) return;
      textViewRef.navigate(hit.charStart, hit.charEnd);
      search.pendingNav = null;
    }
  }

  // Registered on appState (see header note). `jumpToResult` has already set the
  // active doc + `pendingNav`; here we wait for the viewer to mount/load, then
  // retry until the navigation is consumed.
  async function drainNav(_hit: ProjectHit) {
    await tick();
    for (let i = 0; i < 60 && search.pendingNav; i++) {
      tryNavigate();
      if (search.pendingNav) await new Promise((r) => setTimeout(r, 50));
    }
  }

  onMount(async () => {
    search.requestNavigate = drainNav;
    const params = new URLSearchParams(window.location.search);
    const projectId = params.get("id") ?? "";
    if (!projectId) {
      goto("/");
      return;
    }
    // A query passed from the manage launcher (`&q=`) runs immediately.
    const initialQuery = params.get("q") ?? "";
    await loadSearch(projectId);
    if (search.docs.length === 0 && !appState.searchIndexing) {
      // Nothing indexed and nothing in flight — back to manage to add files.
      goto(`/project?id=${encodeURIComponent(projectId)}`);
      return;
    }
    if (initialQuery.trim()) {
      runSearch(initialQuery);
    }
  });
</script>

<main class="flex flex-col h-full" style="background: var(--color-bg); color: var(--color-text);">
  <header
    class="flex flex-col border-b-4 py-3 px-6 gap-2"
    style="border-color: var(--color-border);"
  >
    <nav class="flex items-center gap-1.5 text-sm" style="color: var(--color-text-muted);">
      <a href="/" class="hover:underline">Projects</a>
      <span aria-hidden="true">→</span>
      <a
        href={`/project?id=${encodeURIComponent(search.projectId)}`}
        class="font-medium hover:underline truncate"
        style="color: var(--color-text);">{appState.searchProjectName}</a
      >
    </nav>
    <div class="flex items-center gap-4">
      <h1 class="shrink-0"><SemantraLogo class="h-[1.875rem]" /></h1>
      <SearchBar />
    </div>
  </header>

  {#if appState.searchIndexing}
    <IndexProgressBar active={search.status.active} pending={appState.searchPending} />
  {/if}

  <article class="flex flex-1 flex-row relative items-stretch min-h-0">
    <SearchResults />

    <div class="flex flex-col flex-1 min-w-0">
      {#if appState.searchActiveDoc}
        {@const activeDoc = appState.searchActiveDoc}
        <TabBar />
        {#if activeDoc.filetype === "text"}
          <div class="flex-1 min-h-0">
            <!-- The reader is a pure function of the active document: re-fetch
                 (and remount) per sha512 via `{#key}` + `{#await}`. The await
                 block discards stale fetches when the doc switches, so there's
                 no manual cache state or loader effect to coordinate. -->
            {#key activeDoc.sha512}
              {#await getDocumentText(activeDoc.sha512)}
                <div class="text-sm m-2" style="color: var(--color-text-muted);">Loading…</div>
              {:then text}
                <TextView bind:this={textViewRef} {text} />
              {/await}
            {/key}
          </div>
        {:else if activeDoc.filetype === "csv"}
          <div class="flex-1 min-h-0">
            <!-- Remount per document: canvas-datagrid builds its grid from the
                 fetched data on mount, so a fresh instance per sha512 is cleanest. -->
            {#key activeDoc.sha512}
              <CsvViewer bind:this={csvViewerRef} sha512={activeDoc.sha512} />
            {/key}
          </div>
        {:else}
          <div class="flex-1 min-h-0">
            <!-- Remount on document change: the PDF.js viewer holds a lot of
                 per-document internal state, so a fresh instance is cleaner than
                 reloading in place. -->
            {#key activeDoc.sha512}
              <PdfViewer bind:this={pdfViewerRef} sha512={activeDoc.sha512} />
            {/key}
          </div>
        {/if}
      {:else}
        <div class="text-sm m-2" style="color: var(--color-text-muted);">
          {appState.searchIndexing ? "Indexing documents…" : "No documents."}
        </div>
      {/if}
    </div>
  </article>
</main>
