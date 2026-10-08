//! Modular, document-type-agnostic chunking.
//!
//! Chunking is deliberately decoupled from extraction and embedding so we can
//! experiment with strategies without touching the rest of the pipeline. An
//! extractor turns a file into a canonical full text plus an ordered list of
//! [`Segment`]s (one per page for a PDF, a single segment for plain text, and
//! whatever future file types map onto). A [`Chunker`] then turns segments into
//! [`Chunk`]s, each carrying offsets back into the canonical full text so a
//! search hit can be mapped to its exact source location for highlighting.
//!
//! The default [`TokenWindowChunker`] makes windows of a fixed **token** budget
//! (as counted by the embedding model's tokenizer) **within a single segment**
//! (so a chunk never straddles a page boundary), with a small rewind/overlap
//! between consecutive windows so a passage split across a window edge still
//! appears whole in one chunk. Windows always start and end on word boundaries,
//! so chunk text is a verbatim span of the source.

/// A contiguous run of source text with a known position in the document's
/// canonical full text. Chunkers never merge across segments, so segment
/// boundaries (e.g. PDF pages) are also hard chunk boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The segment's verbatim text.
    pub text: String,
    /// 0-based page index when the source paginates (PDF); `None` for flat text.
    /// For a CSV cell (see [`CellChunker`]) this carries the 0-based data-row
    /// index instead — the reader navigates by (row, col), not by page.
    pub page: Option<usize>,
    /// 0-based column index when the source is tabular (CSV); `None` otherwise.
    /// Only [`CellChunker`] reads it; [`TokenWindowChunker`] ignores it.
    pub col: Option<usize>,
    /// Char offset of this segment's first char within the canonical full text.
    pub base_offset: usize,
}

impl Segment {
    /// A single flat segment covering an entire (non-paginated) document.
    pub fn flat(text: impl Into<String>) -> Self {
        Segment {
            text: text.into(),
            page: None,
            col: None,
            base_offset: 0,
        }
    }
}

/// One unit handed to the embedder.
///
/// Offsets are **char** indices (not bytes), so the frontend — which works in
/// JS string/char space — can slice consistently. `char_start..char_end` indexes
/// the canonical full text (used to highlight in the flat text reader);
/// `page` + `page_char_start` index the originating page's text (used to map
/// onto per-page PDFium character boxes for the PDF reader).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Verbatim text of the window (what gets embedded & displayed). Equals the
    /// canonical full-text slice `[char_start, char_end)` — interior whitespace
    /// is preserved, never normalized.
    pub text: String,
    /// Inclusive char offset of the window's first char in the full text.
    pub char_start: usize,
    /// Exclusive char offset of the window's last char in the full text.
    pub char_end: usize,
    /// Page the chunk came from, if any. For CSV ([`CellChunker`]) this is the
    /// 0-based data-row index instead.
    pub page: Option<usize>,
    /// Char offset of `char_start` relative to its segment/page start. For CSV
    /// ([`CellChunker`]) this carries the 0-based column index instead — the CSV
    /// reader navigates to a cell by (page=row, page_char_start=col).
    pub page_char_start: usize,
}

impl Chunk {
    /// Char length of the source span this chunk covers. Equal in full-text and
    /// page-text space because a chunk never crosses a segment boundary.
    pub fn span_len(&self) -> usize {
        self.char_end - self.char_start
    }
}

/// Turns extracted segments into embeddable chunks. Implementors decide the
/// windowing policy; all share the [`Chunk`] offset contract above.
pub trait Chunker {
    fn chunk(&self, segments: &[Segment]) -> Vec<Chunk>;
}

/// Token-budgeted windows within each segment, snapped to word boundaries.
///
/// `token_starts(text)` returns the byte offset at which each of the model's
/// tokens for `text` begins (no special tokens). Each token is charged to the
/// word it begins in — or, for the model's leading-space pieces (`▁word`), to
/// the word that follows the whitespace. Windows then greedily take whole words
/// while the token sum stays within `size`, and the next window rewinds by up
/// to `overlap` tokens of whole words.
///
/// A single "word" longer than `size` tokens (minified code, base64, a URL
/// wall) is hard-split at token boundaries so no window can exceed the budget.
///
/// With [`sentences`](Self::sentences) on, a window ends at the last sentence
/// boundary that fits the budget (falling back to a word cut only when one
/// sentence alone exceeds it), and the rewind takes whole trailing sentences
/// only — so chunks read as complete thoughts.
pub struct TokenWindowChunker<F: Fn(&str) -> Vec<usize>> {
    /// Max tokens per window (excluding the document prompt and specials).
    pub size: usize,
    /// Tokens of rewind shared between consecutive windows (`< size`).
    pub overlap: usize,
    /// Snap window ends (and rewinds) to sentence boundaries.
    pub sentences: bool,
    token_starts: F,
}

impl<F: Fn(&str) -> Vec<usize>> TokenWindowChunker<F> {
    pub fn new(size: usize, overlap: usize, token_starts: F) -> Self {
        let size = size.max(1);
        // Overlap must be strictly less than size or the window never advances.
        let overlap = overlap.min(size - 1);
        TokenWindowChunker {
            size,
            overlap,
            sentences: false,
            token_starts,
        }
    }

    /// Snap windows to sentence boundaries (see the type docs).
    pub fn snap_to_sentences(mut self) -> Self {
        self.sentences = true;
        self
    }
}

/// Whether a chunk carries any words at all. Chunks of only digits and
/// punctuation (a lone page number, a dot leader) embed to a near-generic
/// vector that scores middlingly against every query, so they're not indexed.
pub fn has_words(text: &str) -> bool {
    text.chars().any(char::is_alphabetic)
}

/// Abbreviations whose trailing period doesn't end a sentence (lower-cased).
const ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "e.g.", "i.e.", "no.", "vol.", "fig.",
    "gen.", "gov.", "sen.", "rep.", "u.s.", "fed.", "crim.", "civ.", "inc.", "co.", "corp.", "ltd.", "jan.",
    "feb.", "mar.", "apr.", "aug.", "sept.", "sep.", "oct.", "nov.", "dec.", "cf.", "al.", "approx.",
];

/// Whether `word` (a whitespace-delimited token) ends a sentence: its last
/// non-closing char is `.`, `!`, `?` or `…` (closing quotes/brackets may
/// follow), and it isn't a known abbreviation or a lone initial like `J.`.
fn ends_sentence(word: &str) -> bool {
    let core = word.trim_end_matches(['"', '\'', '\u{201D}', '\u{2019}', ')', ']', '}']);
    let Some(last) = core.chars().last() else {
        return false;
    };
    if !matches!(last, '.' | '!' | '?' | '\u{2026}') {
        return false;
    }
    if last == '.' {
        let lower = core.to_lowercase();
        let lower = lower.trim_start_matches(['"', '\'', '\u{201C}', '\u{2018}', '(', '[']);
        if ABBREVIATIONS.contains(&lower) {
            return false;
        }
        let mut cs = lower.chars();
        if let (Some(c), Some('.'), None) = (cs.next(), cs.next(), cs.next()) {
            if c.is_alphabetic() {
                return false; // an initial: "J."
            }
        }
    }
    true
}

/// A word and its position within a segment, in both char and byte space (byte
/// positions are kept only to slice the verbatim text cheaply).
struct WordPos {
    char_start: usize,
    char_end: usize,
    byte_start: usize,
    byte_end: usize,
}

/// Split `text` into maximal runs of non-whitespace, recording each word's char
/// and byte span. Mirrors the JS `\s+` split used on the frontend.
fn words_with_offsets(text: &str) -> Vec<WordPos> {
    let mut words = Vec::new();
    let mut char_idx = 0usize;
    let mut start: Option<(usize, usize)> = None; // (char_start, byte_start)
    for (byte, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some((cs, bs)) = start.take() {
                words.push(WordPos {
                    char_start: cs,
                    char_end: char_idx,
                    byte_start: bs,
                    byte_end: byte,
                });
            }
        } else if start.is_none() {
            start = Some((char_idx, byte));
        }
        char_idx += 1;
    }
    if let Some((cs, bs)) = start.take() {
        words.push(WordPos {
            char_start: cs,
            char_end: char_idx,
            byte_start: bs,
            byte_end: text.len(),
        });
    }
    words
}

/// Charge every token start to a word: word `i` owns the starts before its
/// `byte_end` not owned by an earlier word (so a leading-space piece in the gap
/// belongs to the word after it). Words costing more than `max` tokens are
/// split at token starts into pieces of at most `max`. Returns the (possibly
/// split) words with their token costs (each at least 1).
fn costed_words(text: &str, starts: &[usize], max: usize) -> Vec<(WordPos, usize)> {
    let mut out = Vec::new();
    let mut t = 0usize; // next unowned index into `starts`
    for w in words_with_offsets(text) {
        let first = t;
        while t < starts.len() && starts[t] < w.byte_end {
            t += 1;
        }
        let owned = &starts[first..t];
        if owned.len() <= max {
            out.push((w, owned.len().max(1)));
            continue;
        }
        // Hard-split an oversized word every `max` tokens. Cut points are
        // clamped into the word and kept on char boundaries so each piece is a
        // verbatim, non-empty slice.
        let cuts = owned
            .iter()
            .step_by(max)
            .skip(1)
            .map(|&b| b.clamp(w.byte_start, w.byte_end))
            .filter(|&b| text.is_char_boundary(b));
        let (mut piece_start, mut piece_char, mut charged) = (w.byte_start, w.char_start, 0);
        for b in cuts.chain(std::iter::once(w.byte_end)) {
            if b <= piece_start {
                continue;
            }
            let chars = text[piece_start..b].chars().count();
            // Cumulative so the first piece also pays for the leading-space
            // token that starts in the gap before the word.
            let upto = owned.iter().filter(|&&s| s < b).count();
            let cost = upto - charged;
            charged = upto;
            out.push((
                WordPos {
                    char_start: piece_char,
                    char_end: piece_char + chars,
                    byte_start: piece_start,
                    byte_end: b,
                },
                cost.max(1),
            ));
            piece_start = b;
            piece_char += chars;
        }
    }
    out
}

impl<F: Fn(&str) -> Vec<usize>> Chunker for TokenWindowChunker<F> {
    fn chunk(&self, segments: &[Segment]) -> Vec<Chunk> {
        let mut out = Vec::new();
        for seg in segments {
            let words = costed_words(&seg.text, &(self.token_starts)(&seg.text), self.size);
            let n = words.len();
            if n == 0 {
                continue;
            }
            // cum[i] = tokens in words[..i]
            let mut cum = Vec::with_capacity(n + 1);
            cum.push(0usize);
            for (_, c) in &words {
                cum.push(cum[cum.len() - 1] + c);
            }
            // boundary[i]: a sentence ends after word i — by punctuation, or at
            // a paragraph break (a blank line) before the next word. Only the
            // last piece of a hard-split word can end one (pieces of a word
            // have no gap between them, and their text rarely ends in `.`).
            let boundary: Vec<bool> = (0..n)
                .map(|i| {
                    let w = &words[i].0;
                    let gap_end = words.get(i + 1).map_or(seg.text.len(), |(nw, _)| nw.byte_start);
                    let gap = &seg.text[w.byte_end..gap_end];
                    ends_sentence(&seg.text[w.byte_start..w.byte_end])
                        || gap.matches('\n').count() >= 2
                })
                .collect();
            let mut start = 0usize;
            loop {
                // Largest end keeping the window within budget (>= 1 word).
                let mut end = start + 1;
                while end < n && cum[end + 1] - cum[start] <= self.size {
                    end += 1;
                }
                // Snap back to the last sentence end in the window, unless the
                // rest of the segment fits or no sentence ends in its back half
                // (snapping earlier would strand fragments like "R. Crim." or
                // a lone ". ." from a dot leader — fall back to a word cut).
                let mut snapped = false;
                if self.sentences && end < n {
                    let min_fill = self.size / 2;
                    if let Some(e) = (start + 1..=end)
                        .rev()
                        .take_while(|&e| cum[e] - cum[start] >= min_fill)
                        .find(|&e| boundary[e - 1])
                    {
                        end = e;
                        snapped = true;
                    }
                    // Absorb a short tail (a page-number footer, a final short
                    // sentence) rather than leave it as its own chunk, letting
                    // this window run up to a quarter over budget.
                    if cum[n] - cum[end] < self.size / 8 && cum[n] - cum[start] <= self.size + self.size / 4 {
                        end = n;
                    }
                }
                let (first, last) = (&words[start].0, &words[end - 1].0);
                // Verbatim slice from the first word's start to the last word's
                // end: interior whitespace is preserved, so `text` is exactly
                // char_slice(full_text, char_start, char_end).
                out.push(Chunk {
                    text: seg.text[first.byte_start..last.byte_end].to_string(),
                    char_start: seg.base_offset + first.char_start,
                    char_end: seg.base_offset + last.char_end,
                    page: seg.page,
                    page_char_start: first.char_start,
                });
                if end == n {
                    break;
                }
                // Rewind: the earliest next start sharing <= `overlap` tokens
                // with this window, while still advancing at least one word.
                // A sentence-snapped window rewinds by whole sentences only.
                let mut next = end;
                if snapped {
                    if let Some(k) = (start + 1..end).find(|&k| boundary[k - 1] && cum[end] - cum[k] <= self.overlap) {
                        next = k;
                    }
                } else {
                    while next - 1 > start && cum[end] - cum[next - 1] <= self.overlap {
                        next -= 1;
                    }
                }
                start = next;
            }
        }
        out
    }
}

/// One chunk per segment, verbatim and unwindowed — for sources whose segments
/// are already the indexing unit. A CSV maps each cell to a segment, so this
/// turns every cell into exactly one [`Chunk`] (the requirement to "index every
/// field"). The chunk's span covers the whole segment, and `page`/`page_char_start`
/// pass through the segment's `page` (row) and `col` (column) so the grid reader
/// can navigate to the exact cell. Empty segments are dropped (nothing to embed).
#[derive(Clone, Copy, Debug, Default)]
pub struct CellChunker;

impl Chunker for CellChunker {
    fn chunk(&self, segments: &[Segment]) -> Vec<Chunk> {
        segments
            .iter()
            .filter(|s| !s.text.is_empty())
            .map(|s| Chunk {
                char_start: s.base_offset,
                char_end: s.base_offset + s.text.chars().count(),
                text: s.text.clone(),
                page: s.page,
                page_char_start: s.col.unwrap_or(0),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Char-based slice helper so tests assert offsets against the source the
    /// same way the frontend (which indexes by char) will.
    fn char_slice(s: &str, start: usize, end: usize) -> String {
        s.chars().skip(start).take(end - start).collect()
    }

    /// Fake tokenizer: one token per word, starting at the word — makes token
    /// budgets equal word counts.
    fn word_starts(text: &str) -> Vec<usize> {
        words_with_offsets(text).iter().map(|w| w.byte_start).collect()
    }

    /// Fake Gemma-style tokenizer: each word is a `▁word` piece starting at the
    /// preceding space (except the first word), plus a token per extra 4 bytes.
    fn gemma_like(text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for (i, w) in words_with_offsets(text).iter().enumerate() {
            out.push(if i == 0 { w.byte_start } else { w.byte_start - 1 });
            out.extend((w.byte_start + 4..w.byte_end).step_by(4));
        }
        out
    }

    fn windows(size: usize, overlap: usize, text: &str) -> Vec<String> {
        TokenWindowChunker::new(size, overlap, word_starts)
            .chunk(&[Segment::flat(text)])
            .into_iter()
            .map(|c| c.text)
            .collect()
    }

    #[test]
    fn non_overlapping_windows_match_token_budget() {
        assert_eq!(windows(2, 0, "one two three four five"), vec!["one two", "three four", "five"]);
    }

    #[test]
    fn offsets_reconstruct_source_span() {
        let text = "alpha beta gamma delta";
        let chunks = TokenWindowChunker::new(2, 0, word_starts).chunk(&[Segment::flat(text)]);
        assert_eq!(char_slice(text, chunks[0].char_start, chunks[0].char_end), "alpha beta");
        assert_eq!(char_slice(text, chunks[1].char_start, chunks[1].char_end), "gamma delta");
    }

    #[test]
    fn overlap_rewinds_between_windows() {
        // size 3, overlap 1: [a b c], [c d e]; the shared word is "c".
        assert_eq!(windows(3, 1, "a b c d e"), vec!["a b c", "c d e"]);
    }

    #[test]
    fn leading_space_pieces_charge_the_following_word() {
        // Under `gemma_like`: "aaaa" = 1, "bbbbbbbb" = 2, "cc" = 1 tokens.
        let text = "aaaa bbbbbbbb cc";
        let costs: Vec<usize> = costed_words(text, &gemma_like(text), 99).iter().map(|(_, c)| *c).collect();
        assert_eq!(costs, vec![1, 2, 1]);
        let chunks = TokenWindowChunker::new(3, 0, gemma_like).chunk(&[Segment::flat(text)]);
        let texts: Vec<_> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["aaaa bbbbbbbb", "cc"]);
    }

    #[test]
    fn oversized_word_is_hard_split_within_budget() {
        // One 40-byte "word" = 10 tokens under `gemma_like`; budget 4.
        let blob = "x".repeat(40);
        let text = format!("pre {blob} post");
        let chunks = TokenWindowChunker::new(4, 0, gemma_like).chunk(&[Segment::flat(text.clone())]);
        let starts = gemma_like(&text);
        for c in &chunks {
            let (lo, hi) = (text.char_indices().nth(c.char_start).unwrap().0, text.char_indices().nth(c.char_end - 1).unwrap().0 + 1);
            let toks = starts.iter().filter(|&&s| s >= lo && s < hi).count();
            assert!(toks <= 4, "chunk {:?} spans {toks} tokens", c.text);
            assert_eq!(c.text, char_slice(&text, c.char_start, c.char_end));
        }
        let rebuilt: String = chunks.iter().map(|c| c.text.replace(' ', "")).collect();
        assert_eq!(rebuilt, format!("pre{blob}post"), "pieces cover the blob exactly once");
    }

    #[test]
    fn multibyte_oversized_word_splits_on_char_boundaries() {
        let blob = "é".repeat(30); // 60 bytes; gemma_like cuts every 4 bytes
        let chunks = TokenWindowChunker::new(3, 0, gemma_like).chunk(&[Segment::flat(blob.clone())]);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.iter().map(|c| c.text.as_str()).collect::<String>(), blob);
    }

    #[test]
    fn never_crosses_segment_or_page_boundary() {
        let p0 = Segment {
            text: "one two three".into(),
            page: Some(0),
            col: None,
            base_offset: 0,
        };
        // Page 1 starts after page 0's text plus a one-char separator.
        let p1 = Segment {
            text: "four five".into(),
            page: Some(1),
            col: None,
            base_offset: 14,
        };
        let chunks = TokenWindowChunker::new(10, 0, word_starts).chunk(&[p0, p1]);
        assert_eq!(chunks.len(), 2, "each page yields its own chunk");
        assert_eq!(chunks[0].page, Some(0));
        assert_eq!(chunks[0].text, "one two three");
        assert_eq!(chunks[1].page, Some(1));
        assert_eq!(chunks[1].text, "four five");
        // Global offset uses base_offset; page-relative offset resets per page.
        assert_eq!(chunks[1].char_start, 14);
        assert_eq!(chunks[1].page_char_start, 0);
    }

    #[test]
    fn preserves_interior_whitespace_verbatim() {
        let text = "café\tnaïve\n\n  Σigma";
        let chunks = TokenWindowChunker::new(2, 0, word_starts).chunk(&[Segment::flat(text)]);
        assert_eq!(chunks[0].text, "café\tnaïve");
        assert_eq!(chunks[1].text, "Σigma");
        assert_eq!(chunks[0].text, char_slice(text, chunks[0].char_start, chunks[0].char_end));
        assert_eq!(char_slice(text, chunks[1].char_start, chunks[1].char_end), "Σigma");
    }

    #[test]
    fn empty_and_whitespace_only_segments_yield_nothing() {
        assert!(windows(8, 2, "").is_empty());
        assert!(windows(8, 2, "   \n\t ").is_empty());
    }

    #[test]
    fn overlap_is_clamped_below_size() {
        let c = TokenWindowChunker::new(3, 9, word_starts);
        assert_eq!(c.overlap, 2);
        assert!(c.chunk(&[Segment::flat("a b c d e f g")]).len() >= 2, "window still advances");
    }

    fn sentence_windows(size: usize, overlap: usize, text: &str) -> Vec<String> {
        TokenWindowChunker::new(size, overlap, word_starts)
            .snap_to_sentences()
            .chunk(&[Segment::flat(text)])
            .into_iter()
            .map(|c| c.text)
            .collect()
    }

    #[test]
    fn windows_end_at_sentence_boundaries() {
        // Budget 6 words: "One two three. Four five." is 5; adding "Six" would
        // cut the next sentence, so the window stops at the period.
        let text = "One two three. Four five. Six seven eight nine.";
        assert_eq!(
            sentence_windows(6, 0, text),
            vec!["One two three. Four five.", "Six seven eight nine."]
        );
    }

    #[test]
    fn sentence_rewind_takes_whole_sentences() {
        // Overlap 2 fits the 2-word sentence "Four five." but not a partial one.
        let text = "One two three. Four five. Six seven eight.";
        assert_eq!(
            sentence_windows(5, 2, text),
            vec!["One two three. Four five.", "Four five. Six seven eight."]
        );
        // Overlap 1 fits no whole sentence: no rewind at all.
        assert_eq!(
            sentence_windows(5, 1, text),
            vec!["One two three. Four five.", "Six seven eight."]
        );
    }

    #[test]
    fn overlong_sentence_falls_back_to_word_cut() {
        // An 8-word sentence can't fit a 3-word budget: cut at words until a
        // sentence end fits, then snap again.
        let text = "one two three four five six seven eight. Done and dusted.";
        assert_eq!(
            sentence_windows(3, 0, text),
            vec!["one two three", "four five six", "seven eight.", "Done and dusted."]
        );
    }

    #[test]
    fn abbreviations_initials_and_quotes() {
        assert!(!ends_sentence("Mr."));
        assert!(!ends_sentence("e.g."));
        assert!(!ends_sentence("J."));
        assert!(ends_sentence("it."));
        assert!(ends_sentence("go.\u{201D}"));
        assert!(ends_sentence("why?)"));
        assert!(ends_sentence("well\u{2026}"));
        assert!(!ends_sentence("(score:"));
        // "Mr. Smith" stays together even when the budget would allow a cut there.
        assert_eq!(
            sentence_windows(4, 0, "Hi Mr. Smith left. Bye now."),
            vec!["Hi Mr. Smith left.", "Bye now."]
        );
    }

    #[test]
    fn no_snapping_into_the_front_half() {
        // The only sentence end inside the 8-word window is after word 2, which
        // would strand a fragment; cut at the word budget instead.
        let text = "Header end. one two three four five six seven eight nine ten.";
        assert_eq!(
            sentence_windows(8, 0, text),
            vec!["Header end. one two three four five six", "seven eight nine ten."]
        );
        // Dot leaders: every "." is a sentence end, but none may end a window early.
        let toc = "Introduction . . . . . . 1 Background . . . . 4";
        assert!(sentence_windows(8, 0, toc).iter().all(|w| w.split_whitespace().count() >= 4));
    }

    #[test]
    fn short_tail_is_absorbed() {
        // 16-word budget; a trailing page number (1 word < 16/8) joins the
        // last full window instead of becoming its own chunk.
        let body = "w ".repeat(14) + "end.";
        let text = format!("{body} Next sentence starts here and goes on. 44");
        let out = sentence_windows(16, 0, &text);
        assert!(out.iter().all(|w| w != "44"), "{out:?}");
        assert!(out.last().unwrap().ends_with("44"));
    }

    #[test]
    fn has_words_rejects_numbers_and_punctuation() {
        assert!(!has_words("44"));
        assert!(!has_words(". . ."));
        assert!(!has_words("§ 12, 34 — (5)"));
        assert!(has_words("R. Crim."));
        assert!(has_words("Café 12"));
    }

    #[test]
    fn blank_line_is_a_boundary() {
        let text = "Heading words here\n\nBody one two three.";
        assert_eq!(sentence_windows(5, 0, text), vec!["Heading words here", "Body one two three."]);
    }

    #[test]
    fn sentence_offsets_reconstruct_source_span() {
        let text = "Alpha beta.  Gamma\u{2014}delta? Epsilon zeta eta.\r\nTheta iota.";
        for c in TokenWindowChunker::new(4, 2, word_starts).snap_to_sentences().chunk(&[Segment::flat(text)]) {
            assert_eq!(char_slice(text, c.char_start, c.char_end), c.text);
        }
    }

    fn cell(text: &str, page: usize, col: usize, base_offset: usize) -> Segment {
        Segment {
            text: text.into(),
            page: Some(page),
            col: Some(col),
            base_offset,
        }
    }

    #[test]
    fn cell_chunker_emits_one_verbatim_chunk_per_cell() {
        // base_offset mirrors a full text of composed strings joined by '\n':
        // "city: Paris" (11 chars) + '\n' => row 1 cell starts at 12.
        let segs = vec![cell("city: Paris", 0, 0, 0), cell("country: France", 0, 1, 12)];
        let chunks = CellChunker.chunk(&segs);
        assert_eq!(chunks.len(), 2);
        // Verbatim text, and page/page_char_start carry (row, col).
        assert_eq!(chunks[0].text, "city: Paris");
        assert_eq!(chunks[0].page, Some(0)); // row
        assert_eq!(chunks[0].page_char_start, 0); // col
        assert_eq!((chunks[0].char_start, chunks[0].char_end), (0, 11));
        assert_eq!(chunks[1].page, Some(0));
        assert_eq!(chunks[1].page_char_start, 1);
        assert_eq!(chunks[1].char_start, 12);
        assert_eq!(chunks[1].char_end, 12 + "country: France".chars().count());
    }

    #[test]
    fn cell_chunker_preserves_interior_whitespace_and_skips_empty() {
        let segs = vec![
            cell("", 0, 0, 0),                  // empty value cell → dropped
            cell("notes: line1\nline2", 0, 1, 0), // newline inside the value survives
        ];
        let chunks = CellChunker.chunk(&segs);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "notes: line1\nline2");
        assert_eq!(chunks[0].page_char_start, 1);
    }
}
