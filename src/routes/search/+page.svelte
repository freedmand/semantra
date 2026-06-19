<script lang="ts">
  // The launched project workspace: header + search bar, file-grouped results
  // sidebar with relevance feedback, document tabs, and a reader (ProseMirror
  // for text, PDF.js for PDFs) that highlights the matched passage. Reachable at
  // `/search?id=<uuid>`. A thin progress bar shows while the project is still
  // indexing in the background, and newly-finished documents appear as tabs.
  import { onMount, onDestroy, tick } from "svelte";
  import { goto } from "$app/navigation";
  import {
    listDocuments,
    listProjects,
    searchProject,
    projectStatus,
    onIndexProgress,
    type DocMeta,
    type ProjectHit,
    type Preference,
    type ProjectStatus,
  } from "$lib/project/projectClient";
  import { getDocumentText } from "$lib/project/projectClient";
  import { explainMatches, type Explanation } from "$lib/embedding/search";
  import { parseQuery } from "$lib/project/query";
  import type { UnlistenFn } from "@tauri-apps/api/event";
  import SearchBar from "$lib/project/SearchBar.svelte";
  import SearchResults from "$lib/project/SearchResults.svelte";
  import TabBar from "$lib/project/TabBar.svelte";
  import TextView from "$lib/project/TextView.svelte";
  import PdfViewer from "$lib/project/PdfViewer.svelte";
  import IndexProgressBar from "$lib/project/IndexProgressBar.svelte";
  import SemantraLogo from "$lib/SemantraLogo.svelte";

  const SEARCH_LIMIT = 30;

  let projectId = $state("");
  let projectName = $state("Project");
  let docs = $state<DocMeta[]>([]);
  let activeIndex = $state(0);
  const activeDoc = $derived(docs[activeIndex] ?? null);

  let status = $state<ProjectStatus>({ jobs: [], active: null });
  const pendingCount = $derived(status.jobs.filter((j) => j.status === "pending").length);
  const isIndexing = $derived(status.active !== null || pendingCount > 0);

  let query = $state("");
  let results = $state<ProjectHit[]>([]);
  let explanations = $state<Record<number, Explanation | null>>({});
  // Relevance feedback, keyed by chunk id.
  let preferences = $state<Record<number, { hit: ProjectHit; weight: number }>>({});
  let unsearched = $state(true);

  let textViewRef = $state<any>(null);
  let pdfViewerRef = $state<any>(null);
  let pendingNav: ProjectHit | null = null;
  let runId = 0;
  let unlisten: UnlistenFn | null = null;

  onMount(async () => {
    const params = new URLSearchParams(window.location.search);
    projectId = params.get("id") ?? "";
    if (!projectId) {
      goto("/");
      return;
    }
    // A query passed from the manage launcher (`&q=`) runs immediately.
    const initialQuery = params.get("q") ?? "";
    const [d, s, projects] = await Promise.all([
      listDocuments(projectId),
      projectStatus(projectId),
      listProjects(),
    ]);
    docs = d;
    status = s;
    projectName = projects.find((p) => p.projectId === projectId)?.name ?? "Project";
    if (docs.length === 0 && !isIndexing) {
      // Nothing indexed and nothing in flight — back to manage to add files.
      goto(`/project?id=${encodeURIComponent(projectId)}`);
      return;
    }
    if (initialQuery.trim()) {
      query = initialQuery;
      handleSearch(initialQuery);
    }
    unlisten = await onIndexProgress((ev) => {
      if (ev.projectId !== projectId) return;
      if (ev.kind === "progress") {
        status = {
          ...status,
          active: { sha512: ev.sha512, basename: ev.basename, done: ev.done, total: ev.total },
        };
      } else {
        // A file started/finished/failed: refresh the doc list (new tabs appear
        // as documents finish) and the indexing status.
        refreshIndexing();
      }
    });
  });
  onDestroy(() => unlisten?.());

  async function refreshIndexing() {
    docs = await listDocuments(projectId);
    status = await projectStatus(projectId);
  }

  async function handleSearch(q: string) {
    query = q;
    const prefs: Preference[] = Object.values(preferences)
      .filter((p) => p.weight !== 0)
      .map((p) => ({ text: p.hit.text, weight: p.weight }));
    if (q.trim() === "" && prefs.length === 0) {
      results = [];
      unsearched = true;
      return;
    }
    const myRun = ++runId;
    results = await searchProject(projectId, q, prefs, SEARCH_LIMIT);
    unsearched = false;
    explanations = {};

    // Best-effort highlighting: explain against the semantic part of the query.
    const semantic = parseQuery(q).semantic.map((t) => t.text).join(" ").trim() || q;
    const current = results;
    explainMatches(semantic, current.map((r) => r.text))
      .then((exps) => {
        if (myRun !== runId) return;
        const map: Record<number, Explanation | null> = {};
        current.forEach((r, i) => (map[r.index] = exps[i] ?? null));
        explanations = map;
      })
      .catch((e) => console.warn("explain failed", e));
  }

  function setPreference(hit: ProjectHit, weight: number) {
    if (weight === 0) {
      const next = { ...preferences };
      delete next[hit.index];
      preferences = next;
    } else {
      preferences = { ...preferences, [hit.index]: { hit, weight } };
    }
  }

  function tryNavigate() {
    const hit = pendingNav;
    if (!hit || !activeDoc || activeDoc.sha512 !== hit.sha512) return;
    if (activeDoc.filetype === "pdf") {
      if (!pdfViewerRef) return;
      pdfViewerRef.navigate(hit.page ?? 0, hit.pageCharStart, hit.charEnd - hit.charStart);
      pendingNav = null;
    } else {
      // `textViewRef` only binds once the `{#await}` resolves and `TextView`
      // mounts, so its presence already means the text is loaded.
      if (!textViewRef) return;
      textViewRef.navigate(hit.charStart, hit.charEnd);
      pendingNav = null;
    }
  }

  async function jumpToResult(hit: ProjectHit) {
    const idx = docs.findIndex((d) => d.sha512 === hit.sha512);
    if (idx < 0) return;
    activeIndex = idx;
    pendingNav = hit;
    await tick();
    // The viewers may need a beat to mount / load; retry until consumed.
    for (let i = 0; i < 60 && pendingNav; i++) {
      tryNavigate();
      if (pendingNav) await new Promise((r) => setTimeout(r, 50));
    }
  }
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
        href={`/project?id=${encodeURIComponent(projectId)}`}
        class="font-medium hover:underline truncate"
        style="color: var(--color-text);">{projectName}</a
      >
    </nav>
    <div class="flex items-center gap-4">
      <h1 class="shrink-0"><SemantraLogo class="h-[1.875rem]" /></h1>
      <SearchBar
        bind:value={query}
        {preferences}
        onSearch={handleSearch}
        onClearPreference={(h) => setPreference(h, 0)}
        onSetPreference={setPreference}
      />
    </div>
  </header>

  {#if isIndexing}
    <IndexProgressBar active={status.active} pending={pendingCount} />
  {/if}

  <article class="flex flex-1 flex-row relative items-stretch min-h-0">
    <SearchResults
      {results}
      {preferences}
      {explanations}
      {unsearched}
      activeSha={activeDoc?.sha512 ?? null}
      onNavigate={jumpToResult}
      onSetPreference={setPreference}
    />

    <div class="flex flex-col flex-1 min-w-0">
      {#if activeDoc}
        <TabBar bind:index={activeIndex} {docs} />
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
          {isIndexing ? "Indexing documents…" : "No documents."}
        </div>
      {/if}
    </div>
  </article>
</main>
