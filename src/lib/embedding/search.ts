/**
 * Per-token attribution client: `explain_matches` decomposes a query↔chunk
 * similarity into signed token contributions, which `highlight.ts` turns into
 * rendered, colored spans. (Indexing + search now run per-project via
 * `$lib/project/projectClient`; this module only covers result explanation.)
 */
import { invoke } from "@tauri-apps/api/core";

/**
 * One token's signed share of a query↔chunk cosine similarity. Mirrors the Rust
 * `explain::TokenSpan`. Offsets are **UTF-8 byte** offsets into the chunk text
 * (slice with {@link sliceBytes}, not by JS string index).
 */
export interface TokenSpan {
  /** Byte offset (inclusive start) of this token in the chunk text. */
  start: number;
  /** Byte offset (exclusive end) of this token in the chunk text. */
  end: number;
  /** WordPiece sub-tokens of one word share a `wordId`; `null` for special tokens. */
  wordId: number | null;
  /** Exact additive share of the cosine similarity (signed). */
  score: number;
  /** True for `[CLS]`/`[SEP]` — real mass, but not chunk text; usually not rendered. */
  special: boolean;
}

/**
 * The exact additive decomposition of one similarity score: `total === bias +
 * Σ tokens[i].score`. Mirrors the Rust `explain::Explanation`. This is *raw*
 * data — word grouping, normalization, and color are applied on top (see
 * `highlight.ts`).
 */
export interface Explanation {
  tokens: TokenSpan[];
  /** Constant share attributable to no single token. */
  bias: number;
  /** The cosine similarity this explanation decomposes. */
  total: number;
}

/**
 * Attribute the similarity between `query` and each chunk in `texts` to its
 * tokens, in one batched backend pass. Returns one {@link Explanation} per text,
 * in order. Call this with the visible search hits after a search resolves to
 * drive result highlighting.
 */
export async function explainMatches(
  query: string,
  texts: string[],
): Promise<Explanation[]> {
  if (texts.length === 0) return [];
  return await invoke<Explanation[]>("explain_matches", { query, texts });
}
