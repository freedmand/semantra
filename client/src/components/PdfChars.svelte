<script lang="ts">
  import { onMount } from "svelte";
  import type { File, Offset, PdfChar, PdfPosition } from "../types";

  export let file: File;
  export let pageNumber: number;
  export let position: PdfPosition;
  export let selectedOffset: Offset | null;

  function processChars(chars: PdfChar[]): PdfChar[] {
    const result: PdfChar[] = [];
    let textBank: string[] = [];
    let lastChar: PdfChar | null = null;
    for (const char of chars.slice()) {
      if (char[1] != null) {
        if (textBank.length != 0) {
          for (const text of textBank) {
            const newChar: PdfChar = [text, char[1]];
            result.push(newChar);
          }
          textBank = [];
        }
        result.push(char);
        lastChar = char;
      } else {
        if (lastChar != null) {
          const newChar: PdfChar = [char[0], lastChar[1]];
          result.push(newChar);
        } else {
          textBank.push(char[0]);
        }
      }
    }
    return result;
  }

  interface Highlight {
    x: number;
    y: number;
    width: number;
    height: number;
  }

  function getHighlightRange(
    chars: PdfChar[],
    offset: Offset | null
  ): Highlight[] {
    if (offset == null) {
      return [];
    }

    const [start, end] = offset;
    const startCharIndex = position.char_index;
    const endCharIndex = position.char_index + chars.length;
    if (start >= endCharIndex || end <= startCharIndex) {
      return [];
    }

    const highlightedChars = chars.slice(
      Math.max(0, start - startCharIndex),
      Math.min(chars.length, end - startCharIndex)
    );

    // Get smooth highlight rectangles
    const highlights: Highlight[] = [];
    let startChar: PdfChar | null = null;
    let lastChar: PdfChar | null = null;
    const pushRect = () => {
      if (startChar == null || lastChar == null || startChar == lastChar)
        return;
      const x0 = Math.min(startChar[1].x0, lastChar[1].x0);
      const x1 = Math.max(startChar[1].x1, lastChar[1].x1);
      const y0 =
        position.page_height - Math.max(startChar[1].y1, lastChar[1].y1);
      const y1 =
        position.page_height - Math.min(startChar[1].y0, lastChar[1].y0);
      highlights.push({
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
      });
    };

    console.log({ highlightedChars });
    for (const char of highlightedChars) {
      if (lastChar != null) {
        const startCharY = position.page_height - startChar[1].y0;
        const charY = position.page_height - char[1].y1;
        if (charY > startCharY) {
          pushRect();
          startChar = char;
        }
      }
      if (lastChar == null) {
        startChar = char;
      }
      lastChar = char;
    }
    pushRect();

    return highlights;
  }

  let chars: PdfChar[] = [];
  $: processedChars = processChars(chars);
  $: highlights = getHighlightRange(processedChars, selectedOffset);
  $: console.log({ highlights });
  let containerElem: HTMLElement;

  const baseFontSize = 16;
  let mWidth = 10;
  let mHeight = 24;

  onMount(async () => {
    // Measure a monospace 'm'
    const m = document.createElement("span");
    m.textContent = "m";
    m.style.position = "absolute";
    m.style.visibility = "hidden";
    m.style.whiteSpace = "pre";
    m.style.fontFamily = "monospace";
    m.style.fontSize = `${baseFontSize}px`;
    containerElem.appendChild(m);
    const mBounds = m.getBoundingClientRect();
    mWidth = mBounds.width;
    mHeight = mBounds.height;
    containerElem.removeChild(m);

    const response = await fetch(
      `/api/pdfchars?filename=${encodeURIComponent(
        file.filename
      )}&page=${pageNumber}`
    );
    chars = await response.json();
  });
</script>

<div class="absolute left-0 top-0 right-0 bottom-0" bind:this={containerElem}>
  {#each processedChars as char, offset}
    <div
      class="absolute monospace text-transparent"
      style="font-size: {baseFontSize}px; left: {char[1]
        .x0}px; top: {position.page_height -
        char[1]
          .y1}px; width: {mWidth}px; height: {mHeight}px; padding-right: {(position.page_width -
        char[1].x1) /
        ((char[1].x1 - char[1].x0) / mWidth)}px; padding-bottom: {char[1].y0 /
        ((char[1].y1 - char[1].y0) / mHeight)}px;
             transform-origin: top left; transform: scale({(char[1].x1 -
        char[1].x0) /
        mWidth}, {(char[1].y1 - char[1].y0) / mHeight});"
    >
      <span class="whitespace-pre">{char[0]}</span>
    </div>
  {/each}
  {#each highlights as highlight}
    <div
      class="absolute highlight pointer-events-none"
      style="left: {highlight.x}px; top: {highlight.y}px; width: {highlight.width}px; height: {highlight.height}px;"
    />
  {/each}
</div>

<style>
  .highlight {
    background-color: rgb(255 255 0 / 72%);
    mix-blend-mode: darken;
  }
</style>
