<script lang="ts">
  // Root layout: loads the global stylesheet (Tailwind + design tokens) for the
  // whole app, and owns the single app-wide indexing subscription + initial
  // project-list load. Prerender/SSR settings live in +layout.ts (SSR off, so
  // onMount is client-only).
  import "../app.css";
  import { onMount } from "svelte";
  import { initLiveUpdates, refreshProjects } from "$lib/state.svelte";
  import UpdatePrompt from "$lib/UpdatePrompt.svelte";

  let { children } = $props();

  onMount(() => {
    initLiveUpdates();
    refreshProjects();
  });
</script>

{@render children()}
<UpdatePrompt />
