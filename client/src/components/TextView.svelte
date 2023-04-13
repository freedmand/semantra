<script lang="ts">
  import Ace from "ace-builds/src-noconflict/ace";
  import { onMount } from "svelte";
  import type { Navigation } from "../types";

  let editorElement: HTMLDivElement;

  export let text: string;

  // Ace text editor
  let editor;

  export function navigate(start: number, end: number) {
    if (editor) {
      const startPos = editor.session.doc.indexToPosition(start);
      const endPos = editor.session.doc.indexToPosition(end);
      const range = new Ace.Range(
        startPos.row,
        startPos.column,
        endPos.row,
        endPos.column
      );
      editor.selection.setRange(range);
      editor.scrollToLine(
        Math.round((startPos.row + endPos.row) / 2),
        true,
        true,
        () => {}
      );
    }
  }

  onMount(() => {
    editor = Ace.edit(editorElement, {
      mode: "ace/mode/text",
      selectionStyle: "text",
      readOnly: true,
      showFoldWidgets: false,
      showPrintMargin: false,
      showInvisibles: false,
      behavioursEnabled: false,
      vScrollBarAlwaysVisible: true,
      useSoftTabs: false,
    });
    editor.session.setUseWrapMode(true);
    // Set text
    editor.setValue(text);
    editor.selection.setRange(new Ace.Range(0, 0, 0, 0));
  });

  $: {
    if (editor) {
      editor.setValue(text);
      editor.selection.setRange(new Ace.Range(0, 0, 0, 0));
    }
  }
</script>

<div id="editor" bind:this={editorElement} />

<style>
  #editor {
    @apply w-full h-full;
  }

  :global(.myMarker) {
    @apply bg-red-200;
  }

  :global(.ace_invalid) {
    background-color: inherit !important;
    color: inherit !important;
  }

  :global(.ace_indent-guide) {
    background: inherit !important;
  }

  :global(.ace_gutter-cell) {
    @apply text-gray-400;
  }
</style>
