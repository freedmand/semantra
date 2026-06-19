<script lang="ts">
  // Shell for the home + project-manage interfaces: a persistent project sidebar
  // on the left and the routed pane (welcome at "/", manage at "/project") on the
  // right. The sidebar lives here so it stays mounted across navigation between
  // the two — no remount, no flicker, scroll/create state preserved.
  import ProjectSidebar from "$lib/project/ProjectSidebar.svelte";

  let { children } = $props();
</script>

<div
  class="flex h-screen"
  style="background: var(--color-bg); color: var(--color-text);"
>
  <ProjectSidebar />
  <div class="flex-1 min-w-0 flex flex-col">
    {@render children()}
  </div>
</div>
