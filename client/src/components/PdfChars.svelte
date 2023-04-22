<script lang="ts">
  import { onMount, tick } from "svelte";
  import type { File, Offset, PdfChar, PdfPosition } from "../types";
  import { copyChars, layout } from "../layoutEngine";

  export let file: File;
  export let pageNumber: number;
  export let position: PdfPosition;
  export let selectedOffset: Offset | null;
  export let zoom: number;
  export let scrollHighlights: boolean;

  function x(x: number): number {
    return (x / position.page_width) * 100;
  }

  function y(y: number): number {
    return (y / position.page_height) * 100;
  }

  async function scrollHighlightsIntoView(...args: any) {
    await tick();
    const highlights = document.querySelectorAll(".page-highlight");
    if (highlights.length == 0) return;
    highlights[0].scrollIntoView({
      block: "center",
    });
    scrollHighlights = false;
  }

  function processChars(chars: PdfChar[]): [PdfChar[], [number, number][]] {
    // Join words together
    const processedChars: PdfChar[] = [];

    const isSpace = (char: PdfChar): boolean => {
      return /\s/.test(char[0]);
    };

    const wordIndexMap: [number, number][] = [];
    let currentWord: PdfChar[] = [];
    let wordStart: number | null = null;
    let wordEnd: number | null = null;

    const pushChar = (char: PdfChar, start: number, end: number) => {
      processedChars.push(char);
      wordIndexMap.push([start, end]);
    };

    const buildWord = (char: PdfChar, i: number) => {
      if (wordStart == null) wordStart = i;
      wordEnd = i + 1;
      currentWord.push(char);
    };

    const getMin = (l: number[]): number => {
      let min: number;
      for (const x of l) {
        if (min == null || x < min) {
          min = x;
        }
      }
      return min;
    };

    const getMax = (l: number[]): number => {
      let max: number;
      for (const x of l) {
        if (max == null || x > max) {
          max = x;
        }
      }
      return max;
    };

    const pushWord = () => {
      if (currentWord.length == 0 || wordStart == null || wordEnd == null)
        return;
      const word = currentWord.map((x) => x[0]).join("");
      const x0 = getMin(currentWord.map((x) => x[1].x0));
      const x1 = getMax(currentWord.map((x) => x[1].x1));
      const y0 = getMin(currentWord.map((x) => x[1].y0));
      const y1 = getMax(currentWord.map((x) => x[1].y1));
      const char: PdfChar = [word, { x0, x1, y0, y1 }];
      pushChar(char, wordStart, wordEnd);
      currentWord = [];
      wordStart = null;
      wordEnd = null;
    };

    for (let i = 0; i < chars.length; i++) {
      const char = chars[i];
      if (isSpace(char)) {
        pushWord();
        pushChar(char, i, i + 1);
      } else {
        buildWord(char, i);
      }
    }
    pushWord();

    return [processedChars, wordIndexMap];
  }

  let chars: PdfChar[] = [];
  $: words = processChars(copyChars(chars));
  $: processedChars = layout(
    position.page_width,
    position.page_height,
    words[0]
  );

  function getHighlightWordIndices(
    words: [PdfChar[], [number, number][]],
    start: number,
    end: number
  ): number[] {
    const wordIndices = words[1];
    const highlights = wordIndices
      .map<[number, [number, number]]>((x, i) => [i, x])
      .filter((x) => x[1][0] >= start && x[1][1] <= end)
      .map((x) => x[0]);
    return highlights;
  }

  $: highlightWordIndices =
    selectedOffset == null
      ? []
      : getHighlightWordIndices(
          words,
          selectedOffset[0] - position.char_index,
          selectedOffset[1] - position.char_index
        );

  $: scrollHighlights && scrollHighlightsIntoView(highlightWordIndices);
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
    const json = await response.json();
    chars = json.map((x) => [
      x[0],
      {
        x0: x[1][0],
        y0: x[1][1],
        x1: x[1][2],
        y1: x[1][3],
      },
    ]);
  });
</script>

<div class="absolute left-0 top-0 right-0 bottom-0" bind:this={containerElem}>
  {#each processedChars as char, i}
    <div
      class="absolute content-box text-transparent"
      style="left: {(char[1].x0 - (char[1].lpad || 0)) *
        zoom}px; top: {(position.page_height -
        char[1].y1 -
        (char[1].bpad || 0)) *
        zoom}px; width: {(char[1].x1 -
        char[1].x0 +
        (char[1].lpad || 0) +
        (char[1].rpad || 0)) *
        zoom}px; height: {(char[1].y1 -
        char[1].y0 +
        (char[1].bpad || 0) +
        (char[1].tpad || 0)) *
        zoom}px;
        padding-left: {(char[1].lpad || 0) * zoom}px;
        padding-right: {(char[1].rpad || 0) * zoom}px;
        padding-bottom: {(char[1].tpad || 0) * zoom}px;
        padding-top: {(char[1].bpad || 0) * zoom}px;"
    >
      <span
        class="inline-block whitespace-pre origin-top-left select-all"
        class:page-highlight={highlightWordIndices.includes(i)}
        style="font-family: monospace; font-size: {baseFontSize}px; transform: scale({((char[1]
          .x1 -
          char[1].x0) /
          mWidth /
          char[0].length) *
          zoom}, {((char[1].y1 - char[1].y0) / mHeight) * zoom});"
        >{char[0]}</span
      >
    </div>
  {/each}
</div>

<style>
  .page-highlight {
    background-color: rgb(255 255 0 / 72%);
    mix-blend-mode: darken;
  }
</style>
