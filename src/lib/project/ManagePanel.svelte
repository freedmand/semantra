<script lang="ts">
  // The manage view for one project's files, rendered in the main pane of the
  // (manage) shell. Add files (via the dropzone — drop or click), watch indexing
  // progress, rename/delete the project from its title, and launch the search
  // workspace. The parent keys this component on `projectId`, so a sidebar switch
  // remounts it cleanly (fresh onMount load). All state lives in appState.manage;
  // live indexing updates arrive via the root layout's single subscription.
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import {
    appState,
    loadManage,
    addFilesToManage,
    removeManageFile,
    commitManageRename,
    deleteProject,
    manageRowLabel,
    manageRowStats,
    type ManageRow,
  } from "$lib/state.svelte";
  import IndexProgressBar from "$lib/project/IndexProgressBar.svelte";
  import FileDropZone from "$lib/project/FileDropZone.svelte";

  let { projectId }: { projectId: string } = $props();

  const manage = appState.manage;

  onMount(() => loadManage(projectId));

  async function removeFile(row: ManageRow) {
    if (!confirm(`Remove “${row.basename}” from this project? This cannot be undone.`)) {
      return;
    }
    await removeManageFile(row.sha512);
  }

  function startRename() {
    manage.renameText = appState.manageProjectName;
    manage.renaming = true;
  }

  async function removeProject() {
    const count = manage.docs.length;
    const detail =
      count > 0
        ? ` and its ${count.toLocaleString()} document${count === 1 ? "" : "s"}`
        : "";
    if (!confirm(`Delete project “${appState.manageProjectName}”${detail}? This cannot be undone.`)) {
      return;
    }
    await deleteProject(projectId);
    goto("/");
  }

  function launchSearch() {
    goto(`/search?id=${encodeURIComponent(projectId)}`);
  }
</script>

<header
  class="flex items-center justify-between gap-4 py-3 px-6"
>
  {#if manage.renaming}
    <!-- svelte-ignore a11y_autofocus -->
    <input
      class="text-2xl font-mono font-bold px-2 py-1 rounded-md border bg-white min-w-0 max-w-xs"
      style="border-color: var(--color-border); color: var(--color-text);"
      bind:value={manage.renameText}
      autofocus
      onblur={commitManageRename}
      onkeydown={(e) => {
        if (e.key === "Enter") commitManageRename();
        else if (e.key === "Escape") manage.renaming = false;
      }}
    />
  {:else}
    <div class="group flex items-center gap-1 min-w-0">
      <h1 class="text-2xl font-mono font-bold truncate">{appState.manageProjectName}</h1>
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

  {#if appState.manageCanSearch}
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

{#if appState.manageIndexing}
  <IndexProgressBar active={manage.status.active} pending={appState.managePending} />
{/if}

<div class="flex-1 overflow-auto px-6 py-6">
  <div class="flex flex-col gap-4">
    <FileDropZone onFiles={addFilesToManage} busy={manage.adding} />

    {#if manage.error}
      <p class="text-sm" style="color: var(--color-error);">{manage.error}</p>
    {/if}

    {#if !manage.loading && appState.manageRows.length > 0}
      <ul class="flex flex-col gap-1.5">
        {#each appState.manageRows as row (row.sha512)}
          {@const detail = [manageRowLabel(row), manageRowStats(row)].filter(Boolean).join(" · ")}
          <li
            class="rounded-md border px-4 py-3 flex items-center gap-4"
            style="border-color: var(--color-border-soft); background: var(--color-bg-elevated);"
          >
            <div class="flex-1 min-w-0">
              <div class="font-medium break-words">{row.basename}</div>
              <div
                class="text-xs mt-0.5"
                style="color: {row.state === 'error'
                  ? 'var(--color-error)'
                  : 'var(--color-text-muted)'};"
              >
                {detail}
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
