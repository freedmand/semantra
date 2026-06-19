<script lang="ts">
  // The manage view for one project's files, rendered in the main pane of the
  // (manage) shell. Add files (via the dropzone — drop or click), watch indexing
  // progress, rename/delete the project from its title, and launch the search
  // workspace. The parent keys this component on `projectId`, so a sidebar
  // switch remounts it cleanly (fresh onMount load + unlisten).
  import { onMount, onDestroy } from "svelte";
  import { goto } from "$app/navigation";
  import {
    listDocuments,
    projectStatus,
    addFilesToProject,
    deleteFileFromProject,
    renameProject,
    deleteProject,
    onIndexProgress,
    type DocMeta,
    type ProjectStatus,
  } from "$lib/project/projectClient";
  import { projectsState, refreshProjects } from "$lib/project/projectsStore.svelte";
  import type { UnlistenFn } from "@tauri-apps/api/event";
  import IndexProgressBar from "$lib/project/IndexProgressBar.svelte";
  import FileDropZone from "$lib/project/FileDropZone.svelte";

  let { projectId }: { projectId: string } = $props();

  // Name comes from the shared store so a rename here (or in the sidebar) keeps
  // both in sync.
  const projectName = $derived(
    projectsState.list.find((p) => p.projectId === projectId)?.name ?? "Project",
  );

  let docs = $state<DocMeta[]>([]);
  let status = $state<ProjectStatus>({ jobs: [], active: null });
  let loading = $state(true);
  let adding = $state(false);
  let error = $state("");

  // Inline title rename.
  let renaming = $state(false);
  let renameText = $state("");

  let unlisten: UnlistenFn | null = null;

  // Merge committed documents with queued/failed/in-flight jobs into one list.
  type Row = {
    sha512: string;
    basename: string;
    state: "indexed" | "indexing" | "queued" | "error";
    done?: number;
    total?: number;
    error?: string;
  };
  const rows = $derived.by<Row[]>(() => {
    const out: Row[] = docs.map((d) => ({
      sha512: d.sha512,
      basename: d.basename,
      state: "indexed",
    }));
    const activeSha = status.active?.sha512;
    for (const j of status.jobs) {
      if (j.status === "error") {
        out.push({ sha512: j.sha512, basename: j.basename, state: "error", error: j.error });
      } else if (j.sha512 === activeSha) {
        out.push({
          sha512: j.sha512,
          basename: j.basename,
          state: "indexing",
          done: status.active!.done,
          total: status.active!.total,
        });
      } else {
        out.push({ sha512: j.sha512, basename: j.basename, state: "queued" });
      }
    }
    return out;
  });

  const pendingCount = $derived(
    status.jobs.filter((j) => j.status === "pending").length,
  );
  const isIndexing = $derived(status.active !== null || pendingCount > 0);
  const canSearch = $derived(docs.length > 0 || isIndexing);

  async function refresh() {
    if (!projectId) return;
    try {
      const [d, s] = await Promise.all([
        listDocuments(projectId),
        projectStatus(projectId),
        refreshProjects(),
      ]);
      docs = d;
      status = s;
      error = "";
    } catch (e) {
      error = `${e}`;
    } finally {
      loading = false;
    }
  }

  onMount(async () => {
    await refresh();
    unlisten = await onIndexProgress((ev) => {
      if (ev.projectId !== projectId) return;
      if (ev.kind === "progress") {
        // Live-update the in-flight file's chunk progress without a refetch.
        status = {
          ...status,
          active: {
            sha512: ev.sha512,
            basename: ev.basename,
            done: ev.done,
            total: ev.total,
          },
        };
      } else {
        // started / fileDone / error change the file set — reload.
        refresh();
      }
    });
  });
  onDestroy(() => unlisten?.());

  async function removeFile(row: Row) {
    if (!confirm(`Remove “${row.basename}” from this project? This cannot be undone.`)) {
      return;
    }
    await deleteFileFromProject(projectId, row.sha512);
    await refresh();
  }

  function startRename() {
    renameText = projectName;
    renaming = true;
  }

  async function commitRename() {
    const name = renameText.trim();
    renaming = false;
    if (!name || name === projectName) return;
    await renameProject(projectId, name);
    await refreshProjects();
  }

  async function removeProject() {
    const detail =
      docs.length > 0
        ? ` and its ${docs.length} document${docs.length === 1 ? "" : "s"}`
        : "";
    if (!confirm(`Delete project “${projectName}”${detail}? This cannot be undone.`)) {
      return;
    }
    await deleteProject(projectId);
    await refreshProjects();
    goto("/");
  }

  function launchSearch() {
    goto(`/search?id=${encodeURIComponent(projectId)}`);
  }

  // Files chosen from the dropzone (drop or click) are added to this project.
  async function handleFiles(paths: string[]) {
    if (adding) return;
    adding = true;
    try {
      await addFilesToProject(projectId, paths);
      await refresh();
    } catch (e) {
      error = `${e}`;
    } finally {
      adding = false;
    }
  }

  function stateLabel(row: Row): string {
    switch (row.state) {
      case "indexed":
        return "Indexed";
      case "indexing":
        return row.total && row.total > 0
          ? `Indexing… ${row.done}/${row.total}`
          : "Indexing…";
      case "queued":
        return "Queued";
      case "error":
        return `Failed: ${row.error}`;
    }
  }
</script>

<header
  class="flex items-center justify-between gap-4 border-b-4 py-3 px-6"
  style="border-color: var(--color-border);"
>
  {#if renaming}
    <!-- svelte-ignore a11y_autofocus -->
    <input
      class="text-2xl font-mono font-bold px-2 py-1 rounded-md border bg-white min-w-0 max-w-xs"
      style="border-color: var(--color-border); color: var(--color-text);"
      bind:value={renameText}
      autofocus
      onblur={commitRename}
      onkeydown={(e) => {
        if (e.key === "Enter") commitRename();
        else if (e.key === "Escape") renaming = false;
      }}
    />
  {:else}
    <div class="group flex items-center gap-1 min-w-0">
      <h1 class="text-2xl font-mono font-bold truncate">{projectName}</h1>
      <div
        class="shrink-0 flex items-center opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <button
          class="px-1.5 py-1 rounded text-base"
          style="color: var(--color-text-muted);"
          title="Rename project"
          onclick={startRename}
          aria-label="Rename project">✎</button
        >
        <button
          class="px-1.5 py-1 rounded text-base"
          style="color: var(--color-error);"
          title="Delete project"
          onclick={removeProject}
          aria-label="Delete project">🗑</button
        >
      </div>
    </div>
  {/if}

  {#if canSearch}
    <button
      class="px-4 py-2 rounded-md font-medium text-white shrink-0 flex items-center gap-2"
      style="background: var(--color-accent);"
      onclick={launchSearch}
      title="Open search"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
        <circle cx="11" cy="11" r="7" />
        <path d="m21 21-4.3-4.3" stroke-linecap="round" />
      </svg>
      Search
    </button>
  {/if}
</header>

{#if isIndexing}
  <IndexProgressBar active={status.active} pending={pendingCount} />
{/if}

<div class="flex-1 overflow-auto px-6 py-6">
  <div class="flex flex-col gap-4">
    <FileDropZone onFiles={handleFiles} busy={adding} />

    {#if error}
      <p class="text-sm" style="color: var(--color-error);">{error}</p>
    {/if}

    {#if !loading && rows.length > 0}
      <ul class="flex flex-col gap-1.5">
        {#each rows as row (row.sha512)}
          <li
            class="rounded-md border px-4 py-3 flex items-center gap-4"
            style="border-color: var(--color-border-soft); background: var(--color-bg-elevated);"
          >
            <div class="flex-1 min-w-0">
              <div class="font-medium truncate">{row.basename}</div>
              <div
                class="text-xs mt-0.5"
                style="color: {row.state === 'error'
                  ? 'var(--color-error)'
                  : 'var(--color-text-muted)'};"
              >
                {stateLabel(row)}
              </div>
            </div>
            <button
              class="px-2 py-1 rounded text-sm shrink-0"
              style="color: var(--color-text-muted);"
              onclick={() => removeFile(row)}
              title="Remove file">Remove</button
            >
          </li>
        {/each}
      </ul>
    {/if}
  </div>
</div>
