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
//! The default [`WordWindowChunker`] makes fixed-size word windows **within a
//! single segment** (so a chunk never straddles a page boundary) with a small
//! rewind/overlap between consecutive windows, so a passage split across a
//! window edge still appears whole in one chunk.

/// A contiguous run of source text with a known position in the document's
/// canonical full text. Chunkers never merge across segments, so segment
/// boundaries (e.g. PDF pages) are also hard chunk boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The segment's verbatim text.
    pub text: String,
    /// 0-based page index when the source paginates (PDF); `None` for flat text.
    pub page: Option<usize>,
    /// Char offset of this segment's first char within the canonical full text.
    pub base_offset: usize,
}

impl Segment {
    /// A single flat segment covering an entire (non-paginated) document.
    pub fn flat(text: impl Into<String>) -> Self {
        Segment {
            text: text.into(),
            page: None,
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
    /// Whitespace-normalized text of the window (what gets embedded & displayed).
    pub text: String,
    /// Inclusive char offset of the window's first char in the full text.
    pub char_start: usize,
    /// Exclusive char offset of the window's last char in the full text.
    pub char_end: usize,
    /// Page the chunk came from, if any.
    pub page: Option<usize>,
    /// Char offset of `char_start` relative to its segment/page start.
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

/// Fixed word-count windows within each segment, with a configurable rewind
/// (overlap) between windows.
#[derive(Clone, Copy, Debug)]
pub struct WordWindowChunker {
    /// Words per window.
    pub size: usize,
    /// Words of rewind shared between consecutive windows (`< size`).
    pub overlap: usize,
}

impl WordWindowChunker {
    pub fn new(size: usize, overlap: usize) -> Self {
        let size = size.max(1);
        // Overlap must be strictly less than size or the window never advances.
        let overlap = overlap.min(size - 1);
        WordWindowChunker { size, overlap }
    }
}

impl Default for WordWindowChunker {
    fn default() -> Self {
        // 70-word windows with an 8-word rewind: matches the default chunk size
        // while adding light overlap so passages near a window edge still surface
        // whole.
        WordWindowChunker::new(70, 8)
    }
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

impl Chunker for WordWindowChunker {
    fn chunk(&self, segments: &[Segment]) -> Vec<Chunk> {
        let step = (self.size - self.overlap).max(1);
        let mut out = Vec::new();

        for seg in segments {
            let words = words_with_offsets(&seg.text);
            let n = words.len();
            if n == 0 {
                continue;
            }
            let mut start = 0usize;
            loop {
                let end = (start + self.size).min(n);
                let first = &words[start];
                let last = &words[end - 1];
                // Verbatim slice → whitespace-normalized join for embedding/display.
                let text = words[start..end]
                    .iter()
                    .map(|w| &seg.text[w.byte_start..w.byte_end])
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push(Chunk {
                    text,
                    char_start: seg.base_offset + first.char_start,
                    char_end: seg.base_offset + last.char_end,
                    page: seg.page,
                    page_char_start: first.char_start,
                });
                if end == n {
                    break;
                }
                start += step;
            }
        }
        out
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

    #[test]
    fn non_overlapping_windows_match_word_size() {
        let seg = Segment::flat("one two three four five");
        let chunks = WordWindowChunker::new(2, 0).chunk(&[seg]);
        let texts: Vec<_> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["one two", "three four", "five"]);
    }

    #[test]
    fn offsets_reconstruct_source_span() {
        let text = "alpha beta gamma delta";
        let chunks = WordWindowChunker::new(2, 0).chunk(&[Segment::flat(text)]);
        // The full-text span [char_start,char_end) must cover exactly the window's
        // words verbatim (inter-word whitespace included).
        assert_eq!(char_slice(text, chunks[0].char_start, chunks[0].char_end), "alpha beta");
        assert_eq!(char_slice(text, chunks[1].char_start, chunks[1].char_end), "gamma delta");
    }

    #[test]
    fn overlap_rewinds_between_windows() {
        let chunks = WordWindowChunker::new(3, 1).chunk(&[Segment::flat("a b c d e")]);
        let texts: Vec<_> = chunks.iter().map(|c| c.text.as_str()).collect();
        // step = size - overlap = 2: [a b c], [c d e]; the shared word is "c".
        assert_eq!(texts, vec!["a b c", "c d e"]);
        assert_eq!(chunks[0].char_end, 5); // "a b c"
        assert_eq!(chunks[1].char_start, 4); // overlaps into "c"
    }

    #[test]
    fn never_crosses_segment_or_page_boundary() {
        let p0 = Segment {
            text: "one two three".into(),
            page: Some(0),
            base_offset: 0,
        };
        // Page 1 starts after page 0's text plus a one-char separator.
        let p1 = Segment {
            text: "four five".into(),
            page: Some(1),
            base_offset: 14,
        };
        let chunks = WordWindowChunker::new(10, 0).chunk(&[p0, p1]);
        assert_eq!(chunks.len(), 2, "each page yields its own chunk");
        assert_eq!(chunks[0].page, Some(0));
        assert_eq!(chunks[0].text, "one two three");
        assert_eq!(chunks[0].page_char_start, 0);
        assert_eq!(chunks[1].page, Some(1));
        assert_eq!(chunks[1].text, "four five");
        // Global offset uses base_offset; page-relative offset resets per page.
        assert_eq!(chunks[1].char_start, 14);
        assert_eq!(chunks[1].page_char_start, 0);
    }

    #[test]
    fn handles_unicode_and_irregular_whitespace() {
        let text = "café\tnaïve\n\n  Σigma";
        let chunks = WordWindowChunker::new(2, 0).chunk(&[Segment::flat(text)]);
        assert_eq!(chunks[0].text, "café naïve");
        assert_eq!(chunks[1].text, "Σigma");
        // Offsets are char-based: the second window starts at "Σ".
        assert_eq!(char_slice(text, chunks[1].char_start, chunks[1].char_end), "Σigma");
    }

    #[test]
    fn empty_and_whitespace_only_segments_yield_nothing() {
        assert!(WordWindowChunker::default().chunk(&[Segment::flat("")]).is_empty());
        assert!(WordWindowChunker::default()
            .chunk(&[Segment::flat("   \n\t ")])
            .is_empty());
    }

    #[test]
    fn overlap_is_clamped_below_size() {
        // overlap >= size would stall; constructor clamps it.
        let c = WordWindowChunker::new(3, 9);
        assert_eq!(c.overlap, 2);
        let chunks = c.chunk(&[Segment::flat("a b c d e f g")]);
        assert!(chunks.len() >= 2, "window still advances");
    }
}
