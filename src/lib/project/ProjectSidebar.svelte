<script lang="ts">
  // Persistent project navigation sidebar for the (manage) shell. Lists every
  // project (with live indexing badges), creates new ones, and renames/deletes
  // existing ones. The active project is derived from the URL's `?id=`, so the
  // highlight tracks navigation without any local routing state. The project
  // list comes from the shared projectsStore so renames/deletes performed from
  // the manage title stay in sync here. Mirrors the stark black border-r-4
  // sidebar idiom from SearchResults.svelte.
  import { onMount, onDestroy } from "svelte";
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import {
    createProject,
    renameProject,
    deleteProject,
    onIndexProgress,
    type ProjectListItem,
  } from "$lib/project/projectClient";
  import { projectsState, refreshProjects } from "$lib/project/projectsStore.svelte";
  import type { UnlistenFn } from "@tauri-apps/api/event";
  import SemantraLogo from "$lib/SemantraLogo.svelte";

  let error = $state("");

  // Create flow: a ＋ New button toggles an inline input.
  let creatingOpen = $state(false);
  let newName = $state("");
  let creating = $state(false);

  // Inline rename: the project being renamed and its editable text.
  let renamingId = $state<string | null>(null);
  let renameText = $state("");

  // Active project from the URL — highlights the matching row.
  const activeId = $derived(page.url.searchParams.get("id"));

  let unlisten: UnlistenFn | null = null;

  async function refresh() {
    try {
      await refreshProjects();
      error = "";
    } catch (e) {
      error = `${e}`;
    }
  }

  onMount(async () => {
    await refresh();
    // Indexing happens in the background; refresh badges as files start/finish.
    unlisten = await onIndexProgress((ev) => {
      if (ev.kind === "started" || ev.kind === "fileDone" || ev.kind === "error") {
        refresh();
      }
    });
  });
  onDestroy(() => unlisten?.());

  async function create(e: Event) {
    e.preventDefault();
    const name = newName.trim();
    if (!name || creating) return;
    creating = true;
    try {
      const id = crypto.randomUUID();
      await createProject(id, name);
      newName = "";
      creatingOpen = false;
      await refresh();
      // Step straight into the new (empty) project to add files.
      goto(`/project?id=${encodeURIComponent(id)}`);
    } catch (e) {
      error = `${e}`;
    } finally {
      creating = false;
    }
  }

  function open(p: ProjectListItem) {
    goto(`/project?id=${encodeURIComponent(p.projectId)}`);
  }

  function startRename(p: ProjectListItem) {
    renamingId = p.projectId;
    renameText = p.name;
  }

  async function commitRename(p: ProjectListItem) {
    const name = renameText.trim();
    renamingId = null;
    if (!name || name === p.name) return;
    await renameProject(p.projectId, name);
    await refresh();
  }

  async function remove(p: ProjectListItem) {
    const detail =
      p.docCount > 0
        ? ` and its ${p.docCount} document${p.docCount === 1 ? "" : "s"}`
        : "";
    if (!confirm(`Delete project “${p.name}”${detail}? This cannot be undone.`)) {
      return;
    }
    await deleteProject(p.projectId);
    await refresh();
    // If we just deleted the project being viewed, return to the welcome pane.
    if (activeId === p.projectId) goto("/");
  }

  // A short status line for a project's indexing state.
  function statusLabel(p: ProjectListItem): string {
    const parts: string[] = [];
    parts.push(`${p.docCount} ${p.docCount === 1 ? "document" : "documents"}`);
    if (p.pendingCount > 0) parts.push(`indexing ${p.pendingCount}…`);
    if (p.errorCount > 0) parts.push(`${p.errorCount} failed`);
    return parts.join(" · ");
  }
</script>

<nav
  class="w-64 shrink-0 flex flex-col border-r-4"
  style="background: var(--color-bg); border-color: var(--color-border);"
>
  <!-- Wordmark only. Mirrors the workspace header: identical wordmark logo
       and a matching border-b-4. The create action lives in the list below. -->
  <header class="px-4 py-3 border-b-4" style="border-color: var(--color-border);">
    <button onclick={() => goto("/")} title="Home">
      <SemantraLogo class="h-[1.875rem]" />
    </button>
  </header>

  <!-- Project list, led by the New-project affordance -->
  <div class="flex-1 overflow-y-auto px-2 py-2 flex flex-col gap-0.5">
    {#if creatingOpen}
      <form class="px-1 pb-1 flex flex-col gap-2" onsubmit={create}>
        <!-- svelte-ignore a11y_autofocus -->
        <input
          class="w-full px-2 py-1.5 rounded-md border bg-white text-sm"
          style="border-color: var(--color-border); color: var(--color-text);"
          placeholder="New project name…"
          bind:value={newName}
          autofocus
          onkeydown={(e) => {
            if (e.key === "Escape") creatingOpen = false;
          }}
        />
        <button
          type="submit"
          class="px-3 py-1.5 rounded-md text-sm font-medium text-white disabled:opacity-50"
          style="background: var(--color-accent);"
          disabled={creating || newName.trim() === ""}
        >
          {creating ? "Creating…" : "Create project"}
        </button>
      </form>
    {:else}
      <button
        class="w-full text-left px-2 py-1.5 rounded-md text-sm font-medium"
        style="color: var(--color-accent);"
        onclick={() => {
          creatingOpen = true;
          newName = "";
        }}
      >
        ＋ New project
      </button>
    {/if}

    {#if error}
      <p class="px-2 pb-1 text-sm" style="color: var(--color-error);">{error}</p>
    {/if}

    {#if !projectsState.loaded}
      <p class="px-2 py-2 text-sm" style="color: var(--color-text-muted);">Loading…</p>
    {:else if projectsState.list.length === 0}
      <p class="px-2 py-2 text-sm" style="color: var(--color-text-muted);">
        No projects yet.
      </p>
    {:else}
      <ul class="flex flex-col gap-0.5">
        {#each projectsState.list as p (p.projectId)}
          {@const isActive = activeId === p.projectId}
          <li
            class="group rounded-md"
            style="background: {isActive ? 'var(--color-bg-elevated)' : 'transparent'};"
          >
            {#if renamingId === p.projectId}
              <!-- svelte-ignore a11y_autofocus -->
              <input
                class="w-full px-2 py-1.5 rounded-md border bg-white text-sm m-0"
                style="border-color: var(--color-border); color: var(--color-text);"
                bind:value={renameText}
                autofocus
                onblur={() => commitRename(p)}
                onkeydown={(e) => {
                  if (e.key === "Enter") commitRename(p);
                  else if (e.key === "Escape") renamingId = null;
                }}
              />
            {:else}
              <div class="flex items-center px-2 py-1.5 gap-1">
                <button
                  class="flex-1 min-w-0 text-left"
                  onclick={() => open(p)}
                >
                  <div class="font-medium truncate text-sm">{p.name}</div>
                  <div class="text-xs mt-0.5 truncate" style="color: var(--color-text-muted);">
                    {statusLabel(p)}
                    {#if p.pendingCount > 0}
                      <span
                        class="inline-block w-2 h-2 rounded-full ml-0.5 align-middle animate-pulse"
                        style="background: var(--color-accent);"
                      ></span>
                    {/if}
                  </div>
                </button>
                <div
                  class="shrink-0 flex items-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <button
                    class="px-1.5 py-1 rounded text-xs"
                    style="color: var(--color-text-muted);"
                    title="Rename"
                    onclick={() => startRename(p)}
                    aria-label="Rename project">✎</button
                  >
                  <button
                    class="px-1.5 py-1 rounded text-xs"
                    style="color: var(--color-error);"
                    title="Delete project"
                    onclick={() => remove(p)}
                    aria-label="Delete project">🗑</button
                  >
                </div>
              </div>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
  </div>
</nav>
