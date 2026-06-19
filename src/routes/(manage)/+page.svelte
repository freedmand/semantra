<script lang="ts">
  // Home pane ("/"): the welcome / empty state shown in the main pane of the
  // (manage) shell when no project is selected. Doubles as a drop-to-start
  // zone — dropping files here spins up a new project (named after the first
  // file) and steps into it.
  import { goto } from "$app/navigation";
  import { createProject, addFilesToProject } from "$lib/project/projectClient";
  import { refreshProjects } from "$lib/project/projectsStore.svelte";
  import FileDropZone from "$lib/project/FileDropZone.svelte";

  let busy = $state(false);

  async function handleFiles(paths: string[]) {
    if (busy) return;
    busy = true;
    try {
      const base = paths[0].split(/[\\/]/).pop() ?? "New project";
      const name = base.replace(/\.[^.\\/]+$/, "") || "New project";
      const id = crypto.randomUUID();
      await createProject(id, name);
      await addFilesToProject(id, paths);
      await refreshProjects();
      goto(`/project?id=${encodeURIComponent(id)}`);
    } finally {
      busy = false;
    }
  }
</script>

<div class="flex-1 flex flex-col items-center justify-center px-6 gap-4">
  <div class="max-w-md text-center flex flex-col gap-1">
    <p class="text-lg font-medium">Start a new project</p>
    <p class="text-sm" style="color: var(--color-text-muted);">
      Add documents to create a project, or select one from the sidebar.
    </p>
  </div>
  <div class="w-full max-w-md">
    <FileDropZone onFiles={handleFiles} {busy} busyLabel="Creating project…" />
  </div>
</div>
