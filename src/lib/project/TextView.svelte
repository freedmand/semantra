<script lang="ts">
  // Read-only document reader built on ProseMirror. The whole document text lives
  // in a single `code_block` node, which preserves newlines AND gives a clean 1:1
  // mapping from source char offset `c` to ProseMirror position `1 + c` — so
  // `navigate(start, end)` can highlight and scroll to an exact source span.
  import { onMount, onDestroy } from "svelte";
  import { Schema } from "prosemirror-model";
  import { EditorState, Plugin, PluginKey, TextSelection } from "prosemirror-state";
  import { EditorView, Decoration, DecorationSet } from "prosemirror-view";

  let { text }: { text: string } = $props();

  let container: HTMLDivElement;
  let view: EditorView | null = null;

  const schema = new Schema({
    nodes: {
      doc: { content: "block+" },
      code_block: {
        content: "text*",
        group: "block",
        code: true,
        defining: true,
        toDOM: () => ["pre", ["code", 0]],
      },
      text: {},
    },
  });

  const hlKey = new PluginKey("highlight");
  const highlightPlugin = new Plugin({
    key: hlKey,
    state: {
      init: () => DecorationSet.empty,
      apply(tr, old) {
        const set = tr.getMeta(hlKey);
        if (set !== undefined) return set as DecorationSet;
        return old.map(tr.mapping, tr.doc);
      },
    },
    props: {
      decorations(state) {
        return hlKey.getState(state);
      },
    },
  });

  function buildState(t: string): EditorState {
    const content = t.length ? [schema.text(t)] : [];
    const doc = schema.node("doc", null, [schema.node("code_block", null, content)]);
    return EditorState.create({ doc, plugins: [highlightPlugin] });
  }

  /** Highlight and scroll to the source span [start, end) (char offsets). */
  export function navigate(start: number, end: number) {
    if (!view) return;
    const size = view.state.doc.content.size;
    // +1: skip the code_block's opening token. Clamp into the document.
    const from = Math.max(1, Math.min(1 + start, size));
    const to = Math.max(from, Math.min(1 + end, size));
    const deco = DecorationSet.create(view.state.doc, [
      Decoration.inline(from, to, { class: "pm-highlight" }),
    ]);
    const tr = view.state.tr
      .setMeta(hlKey, deco)
      .setSelection(TextSelection.create(view.state.doc, from, to));
    view.dispatch(tr);
    // ProseMirror's transaction-level `scrollIntoView()` is unreliable for a
    // read-only view, so scroll the rendered highlight node into view ourselves
    // once the decoration has been painted.
    requestAnimationFrame(() => {
      view?.dom.querySelector(".pm-highlight")?.scrollIntoView({ block: "start" });
    });
  }

  // `text` is immutable for this component's lifetime: the parent fetches the
  // document text up front and mounts a fresh `TextView` per document (keyed on
  // sha512), so the view is built once here — no reactive rebuild needed.
  onMount(() => {
    view = new EditorView(container, {
      state: buildState(text),
      editable: () => false,
    });
  });

  onDestroy(() => view?.destroy());
</script>

<div bind:this={container} class="pm-reader"></div>

<style>
  .pm-reader {
    width: 100%;
    height: 100%;
    overflow: auto;
    background: var(--color-bg-elevated);
  }

  .pm-reader :global(pre) {
    margin: 0;
    padding: 1rem 1.25rem;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    color: var(--color-text);
  }

  .pm-reader :global(.ProseMirror:focus) {
    outline: none;
  }

  .pm-reader :global(.pm-highlight) {
    background: var(--color-page-highlight);
    border-radius: 2px;
    scroll-margin-top: 24px;
  }
</style>
