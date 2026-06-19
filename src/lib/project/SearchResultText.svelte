<script lang="ts">
  // Renders a hit's text. Once its per-token attribution arrives, the relevant
  // spans are tinted (reusing the embedding module's highlight presentation).
  import { toSegments, colorFor, DEFAULT_HIGHLIGHT } from "$lib/embedding/highlight";
  import type { Explanation } from "$lib/embedding/search";

  let { text, explanation = null }: { text: string; explanation?: Explanation | null } =
    $props();

  const segments = $derived(
    explanation ? toSegments(text, explanation, DEFAULT_HIGHLIGHT) : null,
  );
</script>

{#if segments}{#each segments as seg}{#if seg.weight !== 0}<mark
        class="rounded-sm"
        style={`background:${colorFor(seg.weight)}; color: inherit;`}
        title={seg.score.toFixed(3)}>{seg.text}</mark
      >{:else}{seg.text}{/if}{/each}{:else}{text}{/if}
