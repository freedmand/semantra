/**
 * Presentation layer for search-result highlighting.
 *
 * The backend hands us a raw, exact decomposition of each query↔chunk
 * similarity into signed per-token contributions ({@link Explanation}). This
 * module turns that raw data into something renderable, and is the ONLY place
 * the display choices live, so they can change without touching Rust:
 *
 *   - granularity   — aggregate tokens into words / raw tokens / sentences
 *   - faithfulness  — normalize per-chunk ("relative") or on a fixed scale
 *                     comparable across chunks ("absolute")
 *   - color         — {@link colorFor} maps a positive weight to a green
 *                     background (negative contributions are not shown)
 *
 * Token offsets are UTF-8 **byte** offsets, so all slicing goes through a byte
 * array (correct for non-ASCII text, unlike JS string indexing).
 */
import type { Explanation, TokenSpan } from "./search";

/** How tokens are grouped before highlighting. */
export type Granularity = "word" | "token" | "sentence";

/**
 * How contributions are scaled to color intensity.
 *  - `relative`: normalize to the chunk's own strongest span (vivid; not
 *    comparable across results).
 *  - `absolute`: scale by a fixed constant so intensity means the same thing in
 *    every result (comparable; some chunks barely highlight).
 */
export type Faithfulness = "relative" | "absolute";

export interface HighlightOptions {
  granularity: Granularity;
  faithfulness: Faithfulness;
  /** Per-span contribution that maps to full saturation in "absolute" mode. */
  absoluteScale: number;
  /**
   * Number of words each highlight patch spans. We don't paint every word;
   * instead we pick the few strongest positive windows of roughly this width
   * (see {@link toSegments}), so a chunk reads as a couple of sharp green
   * highlights instead of a per-word stipple.
   */
  windowWords: number;
  /**
   * How many highlight patches to show per chunk at most. Only the top windows
   * by positive contribution are kept; everything else stays plain. Negative
   * ("red") contributions are never shown.
   */
  maxHighlights: number;
}

/** Chosen defaults: word-level, vivid per-chunk, a few green patches. */
export const DEFAULT_HIGHLIGHT: HighlightOptions = {
  granularity: "word",
  faithfulness: "relative",
  absoluteScale: 0.05,
  windowWords: 10,
  maxHighlights: 2,
};

/** A contiguous run of chunk text plus its highlight weight. */
export interface Segment {
  text: string;
  /** Normalized signed weight in [-1, 1]; 0 means a plain (un-highlighted) run. */
  weight: number;
  /** The raw, un-normalized contribution behind this run (for tooltips). */
  score?: number;
  /** Set when this run is an exact match for a quoted keyword literal. */
  keyword?: boolean;
}

/** An aggregated group of tokens over a byte range. */
interface Span {
  start: number;
  end: number;
  score: number;
  /** How many model tokens were folded into this span (for the baseline below). */
  tokens: number;
}

/**
 * Below this |bias| we treat the model as having emitted no constant term. Some
 * heads (e.g. mdbr-leaf-mt) have no Dense bias, so the exact decomposition has
 * nothing to absorb the baseline every token shares — all contributions land on
 * the same side of zero and the diverging green/red color collapses to one hue.
 * Models that DO carry a bias (e.g. mdbr-leaf-ir, ~0.13) need no compensation.
 */
const BIAS_EPSILON = 1e-3;

const decoder = new TextDecoder();

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Emit one aggregated span's text as segments, highlighting only its word
 * characters. WordPiece tokenizers split punctuation into their own tokens (each
 * with its own contribution), so without this a comma or period gets its own
 * colored mark. We push leading/trailing non-word characters as plain runs and
 * color only the inner word characters; a span that is *all* punctuation (a
 * standalone `,`/`.`) renders entirely plain. "Word character" is any Unicode
 * letter or number, so internal punctuation (`don't`, `U.S.A`) stays inside the
 * mark.
 */
function pushHighlighted(
  segments: Segment[],
  text: string,
  weight: number,
  score: number,
): void {
  if (!text) return;
  const lead = text.match(/^[^\p{L}\p{N}]+/u)?.[0].length ?? 0;
  if (lead === text.length) {
    // No word characters at all — never highlight pure punctuation.
    segments.push({ text, weight: 0, score: 0 });
    return;
  }
  const trail = text.match(/[^\p{L}\p{N}]+$/u)?.[0].length ?? 0;
  const mid = text.length - trail;
  if (lead > 0)
    segments.push({ text: text.slice(0, lead), weight: 0, score: 0 });
  segments.push({ text: text.slice(lead, mid), weight, score });
  if (trail > 0) segments.push({ text: text.slice(mid), weight: 0, score: 0 });
}

/**
 * Byte offsets where each sentence begins. A boundary is a maximal run of
 * `. ! ?` followed by whitespace. These punctuation/whitespace bytes are all
 * ASCII, which never appears inside a multi-byte UTF-8 sequence, so scanning the
 * byte array directly is safe.
 */
function sentenceStarts(bytes: Uint8Array): number[] {
  const TERM = new Set([0x2e, 0x21, 0x3f]); // . ! ?
  const WS = new Set([0x20, 0x09, 0x0a, 0x0d]); // space tab nl cr
  const starts = [0];
  let i = 0;
  while (i < bytes.length) {
    if (TERM.has(bytes[i])) {
      let j = i;
      while (j < bytes.length && TERM.has(bytes[j])) j++;
      if (j < bytes.length && WS.has(bytes[j])) {
        while (j < bytes.length && WS.has(bytes[j])) j++;
        if (j < bytes.length) starts.push(j);
        i = j;
        continue;
      }
    }
    i++;
  }
  return starts;
}

/** Group the (non-special) tokens into spans per the chosen granularity. */
function aggregate(
  bytes: Uint8Array,
  tokens: TokenSpan[],
  granularity: Granularity,
): Span[] {
  const real = tokens.filter((t) => !t.special && t.end > t.start);

  if (granularity === "token") {
    return real.map((t) => ({
      start: t.start,
      end: t.end,
      score: t.score,
      tokens: 1,
    }));
  }

  const starts = granularity === "sentence" ? sentenceStarts(bytes) : [];
  const groups = new Map<number | string, Span>();

  real.forEach((t, i) => {
    let key: number | string;
    if (granularity === "word") {
      // Sub-word pieces share a wordId; tokens without one stand alone.
      key = t.wordId ?? `t${i}`;
    } else {
      // Largest sentence-start that is still at or before this token.
      let s = 0;
      for (let k = 0; k < starts.length && starts[k] <= t.start; k++) s = k;
      key = s;
    }
    const existing = groups.get(key);
    if (existing) {
      existing.start = Math.min(existing.start, t.start);
      existing.end = Math.max(existing.end, t.end);
      existing.score += t.score; // contributions are additive
      existing.tokens += 1;
    } else {
      groups.set(key, {
        start: t.start,
        end: t.end,
        score: t.score,
        tokens: 1,
      });
    }
  });

  const sorted = [...groups.values()].sort((a, b) => a.start - b.start);
  if (granularity !== "word") return sorted;

  // Coalesce adjacent spans that aren't separated by whitespace into one whole
  // word. The tokenizer's pre-tokenizer splits on punctuation, so `Alice's`
  // arrives as `Alice` / `'` / `s` with distinct wordIds and would otherwise
  // form three spans — letting a highlight start at a bare `'s`. Merging by the
  // absence of intervening whitespace matches the chunker's word definition.
  const merged: Span[] = [];
  for (const s of sorted) {
    const prev = merged[merged.length - 1];
    const between =
      prev && s.start > prev.end
        ? decoder.decode(bytes.subarray(prev.end, s.start))
        : "";
    if (prev && !/\s/u.test(between)) {
      prev.end = Math.max(prev.end, s.end);
      prev.score += s.score;
      prev.tokens += s.tokens;
    } else {
      merged.push({ ...s });
    }
  }
  return merged;
}

/**
 * Smith-Waterman-style span scoring. A visibly-green span rewards
 * {@link GREEN_BASE} plus its severity (so a stronger green pulls a segment
 * together harder); a plain/gap span costs {@link GAP_PENALTY}, so a segment
 * only stretches across gaps that the surrounding greens can pay for.
 */
const GREEN_BASE = 0.7; // green points lie in [GREEN_BASE, GREEN_BASE + 1]
const GAP_PENALTY = 0.5;

/**
 * A run shorter than {@link MIN_SPAN_WORDS} spans is docked
 * {@link SHORT_WORD_PENALTY} for each word it falls short, so a strong word or
 * two can't win a run alone; the penalty fades to zero once a run is long
 * enough. (Spans are ~one word at "word" granularity, so we count spans.)
 */
const MIN_SPAN_WORDS = 5;
const SHORT_WORD_PENALTY = 0.4;

/**
 * A run longer than {@link MAX_SPAN_WORDS} spans is docked an *increasing*
 * penalty: {@link LONG_WORD_PENALTY} for the 1st word over the limit, twice that
 * for the 2nd, and so on (a triangular sum, ∝ over²). A block therefore stops
 * growing once the marginal green stops paying for the marginal length, keeping
 * a couple of tight blocks instead of one chunk-spanning wash.
 */
const MAX_SPAN_WORDS = 15;
const LONG_WORD_PENALTY = 0.2;

/**
 * Per-span color intensity OUTSIDE the winning runs: the full per-word
 * highlighting is still shown, scaled down to this fraction so the winning
 * blocks clearly dominate.
 */
const MUTED_INTENSITY = 0.3;

/**
 * Smith-Waterman-style local selection over a 1-D sequence of span points.
 * Repeatedly takes the maximum-scoring contiguous run — Kadane's reset to zero
 * is exactly the SW recurrence `H[i] = max(0, H[i-1] + points[i])` — masks it,
 * and repeats, yielding up to `cap` non-overlapping, strictly-positive segments.
 * Each run starts and ends on a positive (green) span, since a leading/trailing
 * penalty never improves the running max, but it absorbs interior gaps the
 * surrounding greens outweigh.
 *
 * A run still shorter than `minLen` spans is scored down by `shortPenalty` for
 * each span it falls short, and a run past `maxLen` is scored down by an
 * increasing `longPenalty` (∝ overshoot²) — together they bias runs toward the
 * `[minLen, maxLen]` band. The accumulation itself stays raw (the reset still
 * keys off the running sum); the penalties only bias which `(start, end)` we
 * record as best. Returns span-index ranges `[start, end)` in document order.
 */
function topScoringSpans(
  points: number[],
  cap: number,
  minLen = 0,
  shortPenalty = 0,
  maxLen = Infinity,
  longPenalty = 0,
): Array<{ start: number; end: number }> {
  const n = points.length;
  const used = new Array<boolean>(n).fill(false);
  const out: Array<{ start: number; end: number }> = [];
  for (let k = 0; k < cap; k++) {
    let best = 0; // require a strictly positive (length-adjusted) run
    let bestStart = -1;
    let bestEnd = -1;
    let cur = 0;
    let curStart = 0;
    for (let i = 0; i < n; i++) {
      if (used[i]) {
        cur = 0;
        curStart = i + 1; // already-claimed spans are hard barriers
        continue;
      }
      if (cur <= 0) {
        cur = points[i];
        curStart = i;
      } else {
        cur += points[i];
      }
      const len = i - curStart + 1;
      const over = Math.max(0, len - maxLen);
      const adjusted =
        cur -
        shortPenalty * Math.max(0, minLen - len) -
        (longPenalty * over * (over + 1)) / 2;
      if (adjusted > best) {
        best = adjusted;
        bestStart = curStart;
        bestEnd = i + 1;
      }
    }
    if (bestStart < 0) break; // nothing positive left
    for (let j = bestStart; j < bestEnd; j++) used[j] = true;
    out.push({ start: bestStart, end: bestEnd });
  }
  return out.sort((a, b) => a.start - b.start);
}

export function toSegments(
  text: string,
  explanation: Explanation,
  options: HighlightOptions = DEFAULT_HIGHLIGHT,
): Segment[] {
  const bytes = new TextEncoder().encode(text);
  const spans = aggregate(bytes, explanation.tokens, options.granularity);

  // When the model emitted no bias term, synthesize the midpoint it omitted: the
  // mean per-token contribution (= cosine / N for this chunk). Subtracting it
  // recenters the spans around zero so coloring shows each word's deviation from
  // the average — restoring green/red. This is the true per-chunk midpoint, which
  // is why it can't be a single hard-coded constant (it scales with chunk length
  // and the chunk's own score). Models with a real bias keep their natural sign.
  const tokenTotal = spans.reduce((n, s) => n + s.tokens, 0);
  const scoreTotal = spans.reduce((a, s) => a + s.score, 0);
  const baseline =
    Math.abs(explanation.bias) < BIAS_EPSILON && tokenTotal > 0
      ? scoreTotal / tokenTotal
      : 0;
  // A span's contribution relative to the baseline (its share of the baseline is
  // proportional to how many tokens it folded in). Only positive contributions
  // are ever highlighted — negatives ("red") are dropped to zero.
  const colorScore = (s: Span) => Math.max(s.score - s.tokens * baseline, 0);

  // Normalized severity in [0, 1]: relative to the chunk's strongest span, or a
  // fixed scale. Drives both the SW scoring and the rendered color intensity.
  let norm: number;
  if (options.faithfulness === "relative") {
    const maxPos = spans.reduce((m, s) => Math.max(m, colorScore(s)), 0);
    norm = maxPos > 0 ? maxPos : 1;
  } else {
    norm = options.absoluteScale > 0 ? options.absoluteScale : 1;
  }
  const severity = (s: Span) => clamp(colorScore(s) / norm, 0, 1);

  // Smith-Waterman selection over the span sequence: visible greens earn points,
  // plain/gap spans bleed them, and we keep only the top few contiguous runs.
  const points = spans.map((s) => {
    const sev = severity(s);
    return sev > 0 ? GREEN_BASE + sev : -GAP_PENALTY;
  });
  const segments = topScoringSpans(
    points,
    options.maxHighlights,
    MIN_SPAN_WORDS,
    SHORT_WORD_PENALTY,
    MAX_SPAN_WORDS,
    LONG_WORD_PENALTY,
  );
  const segByStart = new Map(segments.map((seg) => [seg.start, seg]));

  const results: Segment[] = [];

  let lastSpan = 0;
  const slice = (start: number, end: number) =>
    decoder.decode(bytes.subarray(start, end));

  const flushSpan = (start: number) => {
    if (start > lastSpan) {
      results.push({
        score: 0,
        weight: 0,
        text: slice(lastSpan, start),
      });
      lastSpan = start;
    }
  };

  // Walk the spans. A winning run renders as ONE solid block (gaps and all) at
  // the mean severity of its greens; every other span keeps its own per-word
  // color, just muted, so the winning blocks clearly dominate.
  let i = 0;
  while (i < spans.length) {
    const seg = segByStart.get(i);
    if (seg) {
      const startByte = spans[seg.start].start;
      const endByte = spans[seg.end - 1].end;
      flushSpan(startByte);

      let sum = 0;
      let greens = 0;
      let score = 0;
      for (let j = seg.start; j < seg.end; j++) {
        const sev = severity(spans[j]);
        if (sev > 0) {
          sum += sev;
          greens += 1;
        }
        score += spans[j].score;
      }
      pushHighlighted(
        results,
        slice(startByte, endByte),
        greens > 0 ? sum / greens : 0,
        score,
      );
      lastSpan = endByte;
      i = seg.end;
    } else {
      const span = spans[i];
      flushSpan(span.start);
      pushHighlighted(
        results,
        slice(span.start, span.end),
        severity(span) * MUTED_INTENSITY,
        span.score,
      );
      lastSpan = span.end;
      i += 1;
    }
  }
  flushSpan(text.length);

  return results;
}

/**
 * Turn an {@link Explanation} into an ordered list of {@link Segment}s that
 * together cover the entire `text` (highlighted spans interleaved with plain
 * gaps), ready to render.
 */
export function toSegments2(
  text: string,
  explanation: Explanation,
  options: HighlightOptions = DEFAULT_HIGHLIGHT,
): Segment[] {
  const bytes = new TextEncoder().encode(text);
  const spans = aggregate(bytes, explanation.tokens, options.granularity);

  // When the model emitted no bias term, synthesize the midpoint it omitted: the
  // mean per-token contribution (= cosine / N for this chunk). Subtracting it
  // recenters the spans around zero so coloring shows each word's deviation from
  // the average — restoring green/red. This is the true per-chunk midpoint, which
  // is why it can't be a single hard-coded constant (it scales with chunk length
  // and the chunk's own score). Models with a real bias keep their natural sign.
  const tokenTotal = spans.reduce((n, s) => n + s.tokens, 0);
  const scoreTotal = spans.reduce((a, s) => a + s.score, 0);
  const baseline =
    Math.abs(explanation.bias) < BIAS_EPSILON && tokenTotal > 0
      ? scoreTotal / tokenTotal
      : 0;
  // A span's contribution relative to the baseline (its share of the baseline is
  // proportional to how many tokens it folded in). Only positive contributions
  // are ever highlighted — negatives ("red") are dropped to zero.
  const colorScore = (s: Span) => Math.max(s.score - s.tokens * baseline, 0);

  // Normalization divisor: strongest positive span for "relative" (so the best
  // word in this chunk is full saturation), a fixed scale otherwise.
  let norm: number;
  if (options.faithfulness === "relative") {
    const maxPos = spans.reduce((m, s) => Math.max(m, colorScore(s)), 0);
    norm = maxPos > 0 ? maxPos : 1;
  } else {
    norm = options.absoluteScale > 0 ? options.absoluteScale : 1;
  }
  const weightOf = (s: Span) => clamp(colorScore(s) / norm, 0, 1);

  // Pick the few strongest positive windows: slide a fixed-width window over the
  // spans and greedily take the highest-weight non-overlapping ones, up to the
  // cap. Everything outside a chosen window renders plain.
  const windows = selectWindows(
    spans,
    weightOf,
    options.windowWords,
    options.maxHighlights,
  );

  const slice = (start: number, end: number) =>
    decoder.decode(bytes.subarray(start, end));

  const segments: Segment[] = [];
  let cursor = 0;
  for (const win of windows) {
    const wStart = spans[win.start].start;
    const wEnd = spans[win.end - 1].end;
    if (wStart > cursor) {
      segments.push({ text: slice(cursor, wStart), weight: 0, score: 0 });
    }
    const start = Math.max(wStart, cursor); // guard against any overlap
    if (wEnd > start) {
      pushHighlighted(segments, slice(start, wEnd), win.weight, win.score);
    }
    cursor = Math.max(cursor, wEnd);
  }
  if (cursor < bytes.length) {
    segments.push({ text: slice(cursor, bytes.length), weight: 0, score: 0 });
  }
  return segments;
}

/** A chosen highlight window over a contiguous run of spans `[start, end)`. */
interface Window {
  start: number; // first span index (inclusive)
  end: number; // last span index (exclusive)
  weight: number; // mean span weight, for color intensity
  score: number; // Σ raw contribution, for tooltips
}

/**
 * Greedily select up to `cap` non-overlapping, fixed-width windows of spans,
 * each maximizing total positive {@link weightOf}. The window is `width` spans
 * wide (clamped to the chunk length), and only windows with some positive
 * weight are kept — so a chunk with little signal yields fewer (or no) patches
 * rather than padding weak highlights. Returned in document order.
 */
function selectWindows(
  spans: Span[],
  weightOf: (s: Span) => number,
  width: number,
  cap: number,
): Window[] {
  const n = spans.length;
  if (n === 0 || cap <= 0) return [];
  const w = Math.max(1, Math.min(width, n));
  const weights = spans.map(weightOf);

  const taken = new Array<boolean>(n).fill(false);
  const chosen: Window[] = [];
  for (let k = 0; k < cap; k++) {
    let bestStart = -1;
    let bestSum = 0; // require strictly positive — a zero-weight window is noise
    for (let i = 0; i + w <= n; i++) {
      let overlaps = false;
      let sum = 0;
      for (let j = i; j < i + w; j++) {
        if (taken[j]) {
          overlaps = true;
          break;
        }
        sum += weights[j];
      }
      if (!overlaps && sum > bestSum) {
        bestSum = sum;
        bestStart = i;
      }
    }
    if (bestStart < 0) break;
    let sumWeight = 0;
    let sumScore = 0;
    for (let j = bestStart; j < bestStart + w; j++) {
      taken[j] = true;
      sumWeight += weights[j];
      sumScore += spans[j].score;
    }
    chosen.push({
      start: bestStart,
      end: bestStart + w,
      weight: sumWeight / w,
      score: sumScore,
    });
  }

  return chosen.sort((a, b) => a.start - b.start);
}

/**
 * Minimum keyword-prefix length that gets highlighted. Mirrors the backend's
 * `MIN_PREFIX_LEN` (src-tauri/src/query.rs), which is the trigram floor of the
 * n-gram search index: tokens shorter than this can't be matched by the index,
 * so they're dropped from both the search filter and the highlight. Keep in sync.
 */
const MIN_PREFIX_LEN = 3;

/**
 * Split segments at word-*prefix* matches of any quoted keyword literal, flagging
 * the matched word with `keyword: true` (preserving each parent run's
 * weight/score). Matching is word-prefix and case-insensitive, mirroring the
 * backend keyword filter — so `green` flags the whole word `greenhouse` (a word
 * starting with the literal) but not `evergreen` (mid-word). The full matched
 * word is highlighted, not just the typed prefix. Returns the segments unchanged
 * when there are no literals.
 *
 * Matching runs over the **full concatenated text**, not per-segment, so a
 * literal that straddles segment boundaries is still found — e.g. `text:` after
 * `toSegments` has split it into a `text` run and a `:` run (which previously
 * dropped the keyword box once the async explanation re-rendered the snippet).
 * Each match range is then sliced back onto whatever segments it overlaps, so the
 * matched piece inherits its parent run's weight.
 */
export function markKeywords(
  segments: Segment[],
  literals: string[],
): Segment[] {
  const tokens = literals
    .flatMap((l) => l.split(/\s+/))
    .map((t) => t.trim())
    .filter((t) => t.length >= MIN_PREFIX_LEN);
  if (tokens.length === 0) return segments;

  const escaped = tokens.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  // Word-prefix: a Unicode word boundary before the literal, then the literal,
  // then the rest of the word. `\b`/`\w` are ASCII-only even with `u`, so use
  // explicit Unicode word-character classes for the boundary and the tail.
  const re = new RegExp(
    `(?<![\\p{L}\\p{N}_])(?:${escaped.join("|")})[\\p{L}\\p{N}_]*`,
    "giu",
  );

  // Collect match ranges over the whole snippet (segments concatenate to it).
  const full = segments.map((s) => s.text).join("");
  const ranges: Array<[number, number]> = [];
  re.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(full)) !== null) {
    if (m[0].length === 0) {
      re.lastIndex++; // guard against zero-width matches
      continue;
    }
    ranges.push([m.index, m.index + m[0].length]);
  }
  if (ranges.length === 0) return segments;

  // Walk segments alongside the (sorted, non-overlapping) match ranges, slicing
  // each segment into plain pieces and keyword pieces. A range may span several
  // segments, so it's only consumed once it ends within the current segment.
  const out: Segment[] = [];
  let segStart = 0; // offset of the current segment within `full`
  let r = 0; // index of the next range that may touch this segment
  for (const seg of segments) {
    const segEnd = segStart + seg.text.length;
    while (r < ranges.length && ranges[r][1] <= segStart) r++;
    let cursor = segStart; // absolute position consumed so far
    // Slice [cursor, abs) of this segment, in segment-local coordinates.
    const local = (abs: number) => seg.text.slice(cursor - segStart, abs - segStart);
    let ri = r;
    while (ri < ranges.length && ranges[ri][0] < segEnd) {
      const start = Math.max(ranges[ri][0], segStart);
      const end = Math.min(ranges[ri][1], segEnd);
      if (start > cursor) out.push({ ...seg, text: local(start) });
      cursor = start;
      out.push({ ...seg, text: local(end), keyword: true });
      cursor = end;
      if (ranges[ri][1] <= segEnd) ri++;
      else break; // range continues into the next segment
    }
    r = ri;
    if (cursor < segEnd) out.push({ ...seg, text: local(segEnd) });
    segStart = segEnd;
  }

  // A single match split across segments yields adjacent keyword pieces; merge
  // them so the literal renders as one unbroken box (no seam between, say, the
  // `text` run and the `:` run).
  const merged: Segment[] = [];
  for (const s of out) {
    const prev = merged[merged.length - 1];
    if (s.keyword && prev?.keyword) prev.text += s.text;
    else merged.push(s);
  }
  return merged;
}

/**
 * Background tint for a positive explain contribution in the results list. A
 * flat olive wash (`--color-explain-highlight` in app.css); returns
 * `"transparent"` for zero/negative weight, so only positive contributions are
 * highlighted.
 *
 * This explain tint is deliberately distinct from the *viewer* highlight ink —
 * the PDF reader's `HL_COLOR` and the txt/CSV readers' `--color-page-highlight`,
 * both yellow — which is unchanged. The explain tint marks per-word attribution
 * in the results list; the viewers mark the located passage.
 */
export function colorFor(weight: number): string {
  return weight > 0 ? "rgb(154 134 0 / 18%)" : "transparent";
}
