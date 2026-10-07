<script lang="ts">
  // Renders a hit's text. Once its per-token attribution arrives, the relevant
  // spans are tinted (reusing the embedding module's highlight presentation).
  // Quoted keyword literals from the query are additionally boxed, even before
  // the attribution loads, so exact matches always stand out.
  //
  // Long chunks (~250 tokens) are shown as an excerpt centred on the strongest
  // highlight (or the start, before attribution arrives), with a toggle to
  // expand the whole chunk — so the results list stays scannable.
  import {
    toSegments,
    markKeywords,
    colorFor,
    DEFAULT_HIGHLIGHT,
    type Segment,
  } from "$lib/embedding/highlight";
  import type { Explanation } from "$lib/embedding/search";
  import { appState } from "$lib/state.svelte";

  let {
    text,
    explanation = null,
  }: { text: string; explanation?: Explanation | null } = $props();

  /** Chunks longer than this (chars) are excerpted until expanded. */
  const EXCERPT_CHARS = 420;

  let expanded = $state(false);

  const segments = $derived(
    markKeywords(
      explanation
        ? toSegments(text, explanation, DEFAULT_HIGHLIGHT)
        : [{ text, weight: 0 }],
      appState.search.literals,
    ),
  );

  /** Char window to show: around the heaviest segment, snapped to spaces. */
  const excerpt = $derived.by((): [number, number] | null => {
    if (expanded || text.length <= EXCERPT_CHARS + 80) return null;
    let pos = 0;
    let center = 0;
    let best = 0;
    for (const seg of segments) {
      const w = seg.keyword ? 2 : seg.weight;
      if (w > best) {
        best = w;
        center = pos + seg.text.length / 2;
      }
      pos += seg.text.length;
    }
    let start = Math.max(0, Math.round(center - EXCERPT_CHARS * 0.4));
    let end = Math.min(text.length, start + EXCERPT_CHARS);
    start = Math.max(0, end - EXCERPT_CHARS);
    if (start > 0) {
      const sp = text.indexOf(" ", start);
      if (sp >= 0 && sp < start + 40) start = sp + 1;
    }
    if (end < text.length) {
      const sp = text.lastIndexOf(" ", end);
      if (sp > end - 40) end = sp;
    }
    return [start, end];
  });

  /** The segments clipped to the excerpt window. */
  const shown = $derived.by((): Segment[] => {
    const w = excerpt;
    if (!w) return segments;
    const out: Segment[] = [];
    let pos = 0;
    for (const seg of segments) {
      const s0 = pos;
      const s1 = pos + seg.text.length;
      pos = s1;
      if (s1 <= w[0] || s0 >= w[1]) continue;
      out.push({ ...seg, text: seg.text.slice(Math.max(0, w[0] - s0), Math.min(seg.text.length, w[1] - s0)) });
    }
    return out;
  });

  function toggle(e: Event) {
    e.stopPropagation();
    expanded = !expanded;
  }
</script>

{#if excerpt && excerpt[0] > 0}…{/if}{#each shown as seg}{#if seg.keyword}<mark
      class="rounded-sm font-semibold"
      style="background:rgb(255 224 0); box-shadow: inset 0 0 0 1px #ca8a04; color: inherit;"
      >{seg.text}</mark
    >{:else if seg.weight !== 0}<mark
      class="rounded-sm"
      style={`background:${colorFor(seg.weight)}; color: inherit;`}
      >{seg.text}</mark
    >{:else}{seg.text}{/if}{/each}{#if excerpt && excerpt[1] < text.length}…{/if}
{#if excerpt || expanded}
  <button class="more" onclick={toggle}>{expanded ? "less" : "more"}</button>
{/if}

<style>
  .more {
    margin-left: 0.4em;
    padding: 0 0.35em;
    font-size: 0.75rem;
    color: var(--color-text-muted);
    border-radius: var(--radius-sm);
    text-decoration: underline;
  }
  .more:hover {
    color: var(--color-text);
    background: var(--color-bg-hover);
  }
</style>
