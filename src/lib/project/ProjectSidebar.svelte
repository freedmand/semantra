<script lang="ts">
  // Persistent project navigation sidebar for the (manage) shell. Lists every
  // project (with live indexing badges), creates new ones, and renames/deletes
  // existing ones. The active project is derived from the URL's `?id=`, so the
  // highlight tracks navigation without any local routing state. All app data —
  // the project list, create/rename buffers — lives in the central appState; the
  // project list is loaded and kept fresh (live badges) by the root layout's
  // single indexing subscription, so there's no per-component load/subscribe here.
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import { type ProjectListItem } from "$lib/project/projectClient";
  import {
    appState,
    createProject,
    renameProject,
    deleteProject,
    projectStatusLabel,
  } from "$lib/state.svelte";
  import SemantraLogo from "$lib/SemantraLogo.svelte";

  const sidebar = appState.sidebar;

  // Active project from the URL — highlights the matching row.
  const activeId = $derived(page.url.searchParams.get("id"));

  async function create(e: Event) {
    e.preventDefault();
    const name = sidebar.newName.trim();
    if (!name || sidebar.creating) return;
    sidebar.creating = true;
    try {
      const id = crypto.randomUUID();
      await createProject(id, name);
      sidebar.newName = "";
      sidebar.creatingOpen = false;
      sidebar.error = "";
      // Step straight into the new (empty) project to add files.
      goto(`/project?id=${encodeURIComponent(id)}`);
    } catch (e) {
      sidebar.error = `${e}`;
    } finally {
      sidebar.creating = false;
    }
  }

  function open(p: ProjectListItem) {
    goto(`/project?id=${encodeURIComponent(p.projectId)}`);
  }

  function startRename(p: ProjectListItem) {
    sidebar.renamingId = p.projectId;
    sidebar.renameText = p.name;
  }

  async function commitRename(p: ProjectListItem) {
    const name = sidebar.renameText.trim();
    sidebar.renamingId = null;
    if (!name || name === p.name) return;
    await renameProject(p.projectId, name);
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
    // If we just deleted the project being viewed, return to the welcome pane.
    if (activeId === p.projectId) goto("/");
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
    {#if sidebar.creatingOpen}
      <form class="px-1 pb-1 flex flex-col gap-2" onsubmit={create}>
        <!-- svelte-ignore a11y_autofocus -->
        <input
          class="w-full px-2 py-1.5 rounded-md border bg-white text-sm"
          style="border-color: var(--color-border); color: var(--color-text);"
          placeholder="New project name…"
          bind:value={sidebar.newName}
          autofocus
          onkeydown={(e) => {
            if (e.key === "Escape") sidebar.creatingOpen = false;
          }}
        />
        <button
          type="submit"
          class="px-3 py-1.5 rounded-md text-sm font-medium text-white disabled:opacity-50"
          style="background: var(--color-accent);"
          disabled={sidebar.creating || sidebar.newName.trim() === ""}
        >
          {sidebar.creating ? "Creating…" : "Create project"}
        </button>
      </form>
    {:else}
      <button
        class="w-full text-left px-2 py-1.5 rounded-md text-sm font-medium"
        style="color: var(--color-accent);"
        onclick={() => {
          sidebar.creatingOpen = true;
          sidebar.newName = "";
        }}
      >
        ＋ New project
      </button>
    {/if}

    {#if sidebar.error}
      <p class="px-2 pb-1 text-sm" style="color: var(--color-error);">{sidebar.error}</p>
    {/if}

    {#if !appState.projectsLoaded}
      <p class="px-2 py-2 text-sm" style="color: var(--color-text-muted);">Loading…</p>
    {:else if appState.projects.length === 0}
      <p class="px-2 py-2 text-sm" style="color: var(--color-text-muted);">
        No projects yet.
      </p>
    {:else}
      <ul class="flex flex-col gap-0.5">
        {#each appState.projects as p (p.projectId)}
          {@const isActive = activeId === p.projectId}
          <li
            class="group rounded-md"
            style="background: {isActive ? 'var(--color-bg-elevated)' : 'transparent'};"
          >
            {#if sidebar.renamingId === p.projectId}
              <!-- svelte-ignore a11y_autofocus -->
              <input
                class="w-full px-2 py-1.5 rounded-md border bg-white text-sm m-0"
                style="border-color: var(--color-border); color: var(--color-text);"
                bind:value={sidebar.renameText}
                autofocus
                onblur={() => commitRename(p)}
                onkeydown={(e) => {
                  if (e.key === "Enter") commitRename(p);
                  else if (e.key === "Escape") sidebar.renamingId = null;
                }}
              />
            {:else}
              <div class="flex items-center px-2 py-1.5 gap-1">
                <button
                  class="flex-1 min-w-0 text-left"
                  onclick={() => open(p)}
                >
                  <div class="font-medium text-sm break-words">{p.name}</div>
                  <div class="text-xs mt-0.5 break-words" style="color: var(--color-text-muted);">
                    {projectStatusLabel(p)}
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
