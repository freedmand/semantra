<script lang="ts">
  // A thin, single-line indexing progress bar. Shown while a project still has
  // files being embedded in the background. Used by both the manage page and
  // the (launched) search workspace.
  import type { ActiveStatus } from "./projectClient";

  let {
    active = null,
    pending = 0,
  }: { active?: ActiveStatus | null; pending?: number } = $props();

  // Fraction for the bar; falls back to a thin sliver when totals are unknown.
  const fraction = $derived(
    active && active.total > 0 ? active.done / active.total : 0,
  );
  const label = $derived(
    active
      ? `Indexing ${active.basename}` +
        (active.total > 0
          ? ` — ${active.done.toLocaleString()}/${active.total.toLocaleString()} chunks`
          : "…")
      : "Indexing…",
  );
</script>

<div class="w-full px-4 py-1.5 flex items-center gap-3 text-xs border-b"
  style="border-color: var(--color-border-soft); background: var(--color-bg-elevated); color: var(--color-text-muted);">
  <span class="shrink-0 truncate max-w-[60%]">{label}</span>
  <div
    class="flex-1 h-1 rounded-full overflow-hidden"
    style="background: var(--color-bg-hover);"
  >
    <div
      class="h-full rounded-full transition-[width] duration-200"
      style="width: {Math.max(4, fraction * 100)}%; background: var(--color-accent);"
    ></div>
  </div>
  {#if pending > 0}
    <span class="shrink-0">{pending.toLocaleString()} queued</span>
  {/if}
</div>
