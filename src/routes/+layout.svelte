<script lang="ts">
  // Root layout: loads the global stylesheet (Tailwind + design tokens) for the
  // whole app. Prerender/SSR settings live in +layout.ts.
  import "../app.css";

  let { children } = $props();
</script>

{@render children()}
