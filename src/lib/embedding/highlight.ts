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
 *   - color         — {@link colorFor} maps a signed weight to a diverging
 *                     (green ↑ / red ↓) background
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
   * Max difference in normalized weight (color, ~[0, 2]) between neighbouring
   * highlights for them to fuse into one patch (see {@link consolidate}). Keep
   * it tight so similar-scoring neighbours coalesce while real jumps stay as
   * patch boundaries. `0` disables consolidation.
   */
  consolidateDelta: number;
}

/** Chosen defaults: word-level, vivid per-chunk, diverging color. */
export const DEFAULT_HIGHLIGHT: HighlightOptions = {
  granularity: "word",
  faithfulness: "relative",
  absoluteScale: 0.05,
  consolidateDelta: 0.1,
};

/** A contiguous run of chunk text plus its highlight weight. */
export interface Segment {
  text: string;
  /** Normalized signed weight in [-1, 1]; 0 means a plain (un-highlighted) run. */
  weight: number;
  /** The raw, un-normalized contribution behind this run (for tooltips). */
  score: number;
}

/** An aggregated group of tokens over a byte range. */
interface Span {
  start: number;
  end: number;
  score: number;
}

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
  if (lead > 0) segments.push({ text: text.slice(0, lead), weight: 0, score: 0 });
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
    return real.map((t) => ({ start: t.start, end: t.end, score: t.score }));
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
    } else {
      groups.set(key, { start: t.start, end: t.end, score: t.score });
    }
  });

  return [...groups.values()].sort((a, b) => a.start - b.start);
}

/**
 * Turn an {@link Explanation} into an ordered list of {@link Segment}s that
 * together cover the entire `text` (highlighted spans interleaved with plain
 * gaps), ready to render.
 */
export function toSegments(
  text: string,
  explanation: Explanation,
  options: HighlightOptions = DEFAULT_HIGHLIGHT,
): Segment[] {
  const bytes = new TextEncoder().encode(text);
  const spans = aggregate(bytes, explanation.tokens, options.granularity);

  // Normalization divisor: chunk-max for "relative", a fixed scale otherwise.
  let norm: number;
  if (options.faithfulness === "relative") {
    const maxAbs = spans.reduce((m, s) => Math.max(m, Math.abs(s.score)), 0);
    norm = maxAbs > 0 ? maxAbs : 1;
  } else {
    norm = options.absoluteScale > 0 ? options.absoluteScale : 1;
  }

  const slice = (start: number, end: number) =>
    decoder.decode(bytes.subarray(start, end));

  const segments: Segment[] = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start > cursor) {
      segments.push({ text: slice(cursor, span.start), weight: 0, score: 0 });
    }
    const start = Math.max(span.start, cursor); // guard against any overlap
    if (span.end > start) {
      pushHighlighted(
        segments,
        slice(start, span.end),
        clamp(span.score / norm, -1, 1),
        span.score,
      );
    }
    cursor = Math.max(cursor, span.end);
  }
  if (cursor < bytes.length) {
    segments.push({ text: slice(cursor, bytes.length), weight: 0, score: 0 });
  }
  return consolidate(segments, options.consolidateDelta);
}

/**
 * Fuse neighbouring highlights of similar color into patches.
 *
 * When every word carries some weight, a per-word highlight reads as visual
 * noise — a stipple of slightly different greens. This smooths it by
 * agglomerative clustering on *color distance*: neighbouring spans whose
 * normalized weights are within `maxDelta` collapse into one run that assumes
 * their (length-weighted) average, painting over the whitespace/punctuation
 * between them so the patch is contiguous.
 *
 * It is iterative on purpose. Each round fuses the single closest-in-color
 * adjacent pair, then re-evaluates — because the merged run's new average
 * shifts its distance to *both* neighbours, which can open or close later
 * merges. It repeats until the closest remaining neighbours differ by more than
 * `maxDelta` (the "no more moves" fixpoint), leaving real score jumps as patch
 * boundaries so the result stays dynamic. O(n²) worst case over the highlights
 * in one chunk — trivially fast at these sizes.
 *
 * Returns a fresh segment list; the input is not mutated. A non-positive
 * `maxDelta`, or fewer than two highlights, returns the segments unchanged.
 */
export function consolidate(segments: Segment[], maxDelta: number): Segment[] {
  if (maxDelta <= 0) return segments;

  // One run per highlight, located by character offset in the full text. The
  // length-weighted score sum keeps a merged run's average independent of the
  // order spans were folded in.
  interface Run {
    start: number;
    end: number;
    weighted: number; // Σ weightᵢ · lenᵢ over highlighted chars
    chars: number; // Σ lenᵢ
    score: number; // Σ raw contribution (additive)
  }
  const runs: Run[] = [];
  let pos = 0;
  for (const seg of segments) {
    const len = seg.text.length;
    if (seg.weight !== 0) {
      runs.push({
        start: pos,
        end: pos + len,
        weighted: seg.weight * len,
        chars: len,
        score: seg.score,
      });
    }
    pos += len;
  }
  if (runs.length < 2) return segments;

  const weightOf = (r: Run) => r.weighted / r.chars;

  // Agglomerative merge: fuse the closest-in-color adjacent pair each round
  // until none are within maxDelta.
  let merged = false;
  for (;;) {
    let best = -1;
    let bestDelta = Infinity;
    for (let i = 0; i + 1 < runs.length; i++) {
      const d = Math.abs(weightOf(runs[i]) - weightOf(runs[i + 1]));
      if (d < bestDelta) {
        bestDelta = d;
        best = i;
      }
    }
    if (best < 0 || bestDelta > maxDelta) break;
    const a = runs[best];
    const b = runs[best + 1];
    a.end = b.end; // absorb b and the plain gap between them
    a.weighted += b.weighted;
    a.chars += b.chars;
    a.score += b.score;
    runs.splice(best + 1, 1);
    merged = true;
  }

  if (!merged) return segments; // Nothing was close enough; leave as-is.

  // Repaint: plain gaps between runs, each run at its averaged weight.
  const fullText = segments.map((s) => s.text).join("");
  const out: Segment[] = [];
  let cursor = 0;
  for (const run of runs) {
    if (run.start > cursor) {
      out.push({ text: fullText.slice(cursor, run.start), weight: 0, score: 0 });
    }
    out.push({
      text: fullText.slice(run.start, run.end),
      weight: run.weighted / run.chars,
      score: run.score,
    });
    cursor = run.end;
  }
  if (cursor < fullText.length) {
    out.push({ text: fullText.slice(cursor), weight: 0, score: 0 });
  }
  return out;
}

/** Background alpha at full saturation. */
const MAX_ALPHA = 0.6;
/**
 * Floor alpha for any non-zero contribution. A linear ramp leaves faint spans
 * almost invisible; this guarantees even the weakest real contribution reads.
 */
const MIN_ALPHA = 0.08;
/**
 * Gamma applied to the magnitude before scaling. Values < 1 push low/mid
 * magnitudes up the ramp, so the highlighting reads clearly without being a
 * slight, hard-to-see fade.
 */
const ALPHA_GAMMA = 0.7;

/**
 * Map a normalized signed weight in [-1, 1] to a CSS background color: green for
 * tokens that pushed the similarity up, red for tokens that pulled it down,
 * alpha scaling with magnitude. Returns `"transparent"` for ~zero weight.
 *
 * The magnitude→alpha mapping is intentionally non-linear: a gamma curve lifts
 * weaker contributions and a floor keeps them visible, ramping color in
 * aggressively so highlights are obvious instead of barely-there.
 */
export function colorFor(weight: number): string {
  if (!weight) return "transparent";
  const mag = Math.pow(Math.min(Math.abs(weight), 1), ALPHA_GAMMA);
  const alpha = (MIN_ALPHA + mag * (MAX_ALPHA - MIN_ALPHA)).toFixed(3);
  return weight > 0
    ? `hsl(145 72% 42% / ${alpha})`
    : `hsl(5 78% 51% / ${alpha})`;
}
