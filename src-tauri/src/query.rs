//! Query syntax parsing + relevance-feedback weight normalization.
//!
//! The query box mixes two kinds of terms:
//!
//! - **Semantic** terms (bare text) drive dense vector search, with optional
//!   `+`/`-` weights: `dog`, `dog + cat`, `dog - cat`, `dog +1.2 cat`,
//!   `+3 dogs are nice -2 cats are mean`. Ported from semantra-web's `parseQuery`.
//! - **Literal** terms (double-quoted) are exact keyword filters answered by the
//!   BM25 full-text index: `"dog"`. They are ANDed together and intersected with
//!   the semantic results, so `good "dog"` is *semantic "good" filtered to chunks
//!   containing the keyword "dog"*, and `"dog" "cat"` is *chunks containing both
//!   keywords*.
//!
//! This module is pure (no embedding / DB), so the grammar and the weight math
//! are unit-tested in isolation; `lib.rs` consumes the result.

/// Golden-ratio split between positive and negative term mass (from semantra-web).
pub const POSITIVE_RATIO: f32 = 0.618_033_988_75;

/// A semantic term with its signed weight (pre-normalization).
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedTerm {
    pub text: String,
    pub weight: f32,
}

/// A relevance-feedback preference: the text of a result the user marked, and a
/// signed weight (`+1` relevant, `-1` not). Folded into the same centroid as the
/// semantic query terms.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct Preference {
    pub text: String,
    pub weight: f32,
}

/// A parsed query: weighted semantic terms plus exact-keyword literals.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ParsedQuery {
    pub semantic: Vec<WeightedTerm>,
    pub literals: Vec<String>,
}

/// Parse a raw query string into semantic terms + quoted literals.
///
/// Double-quoted spans become literals (an unterminated quote takes the rest of
/// the string, leniently). The remainder is split on `+`/`-` boundaries into
/// signed/weighted semantic terms, mirroring semantra-web's regex
/// `([+\-]?\d*\.?\d*\s*)?([^+\-]+)`.
pub fn parse_query(raw: &str) -> ParsedQuery {
    let (remainder, literals) = extract_literals(raw);
    let semantic = parse_weighted_terms(&remainder);
    ParsedQuery { semantic, literals }
}

/// Pull `"quoted"` spans out as literals; return (remainder-without-quotes,
/// literals). Each quoted span is trimmed; empty quotes are ignored.
fn extract_literals(raw: &str) -> (String, Vec<String>) {
    let mut remainder = String::new();
    let mut literals = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    for ch in raw.chars() {
        if ch == '"' {
            if in_quote {
                let lit = current.trim();
                if !lit.is_empty() {
                    literals.push(lit.to_string());
                }
                current.clear();
                in_quote = false;
            } else {
                in_quote = true;
            }
        } else if in_quote {
            current.push(ch);
        } else {
            remainder.push(ch);
        }
    }
    // Unterminated quote: treat the trailing content as a literal too.
    if in_quote {
        let lit = current.trim();
        if !lit.is_empty() {
            literals.push(lit.to_string());
        }
    }
    (remainder, literals)
}

/// Split `text` into weighted semantic terms on `+`/`-` boundaries.
fn parse_weighted_terms(text: &str) -> Vec<WeightedTerm> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < n {
        // Skip whitespace between terms.
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }

        // Optional sign.
        let mut sign: Option<char> = None;
        if chars[i] == '+' || chars[i] == '-' {
            sign = Some(chars[i]);
            i += 1;
        }
        // Optional numeric weight (digits and a dot).
        let num_start = i;
        while i < n && (chars[i].is_ascii_digit() || chars[i] == '.') {
            i += 1;
        }
        let num: String = chars[num_start..i].iter().collect();
        // Spaces between the weight and the term text.
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        // Term text runs until the next sign.
        let text_start = i;
        while i < n && chars[i] != '+' && chars[i] != '-' {
            i += 1;
        }
        let term: String = chars[text_start..i].iter().collect();
        let term = term.trim();

        // weight = parsed signed number, else ±1 from the sign (default +1).
        let weight = match num.parse::<f32>() {
            Ok(v) => {
                if sign == Some('-') {
                    -v.abs()
                } else {
                    v
                }
            }
            Err(_) => {
                if sign == Some('-') {
                    -1.0
                } else {
                    1.0
                }
            }
        };

        if !term.is_empty() {
            out.push(WeightedTerm {
                text: term.to_string(),
                weight,
            });
        } else if sign.is_none() && num.is_empty() {
            // No progress (shouldn't happen due to whitespace skip) — guard anyway.
            i += 1;
        }
    }
    out
}

/// Normalize term + preference weights in place so positive mass sums toward
/// `POSITIVE_RATIO` and negative mass toward `1 - POSITIVE_RATIO`, split evenly
/// across the count on each side (semantra-web's `handleSearch`). Relative
/// magnitudes between same-sign terms are preserved.
pub fn normalize_weights(semantic: &mut [WeightedTerm], prefs: &mut [Preference]) {
    let pos = semantic.iter().filter(|t| t.weight > 0.0).count()
        + prefs.iter().filter(|p| p.weight > 0.0).count();
    let neg = semantic.iter().filter(|t| t.weight < 0.0).count()
        + prefs.iter().filter(|p| p.weight < 0.0).count();

    let scale = |w: f32| -> f32 {
        if w > 0.0 && pos > 0 {
            w * POSITIVE_RATIO / pos as f32
        } else if w < 0.0 && neg > 0 {
            w * (1.0 - POSITIVE_RATIO) / neg as f32
        } else {
            w
        }
    };
    for t in semantic.iter_mut() {
        t.weight = scale(t.weight);
    }
    for p in prefs.iter_mut() {
        p.weight = scale(p.weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(pairs: &[(&str, f32)]) -> Vec<WeightedTerm> {
        pairs
            .iter()
            .map(|(t, w)| WeightedTerm {
                text: t.to_string(),
                weight: *w,
            })
            .collect()
    }

    #[test]
    fn bare_term_is_semantic() {
        let p = parse_query("dog");
        assert_eq!(p.semantic, terms(&[("dog", 1.0)]));
        assert!(p.literals.is_empty());
    }

    #[test]
    fn quoted_term_is_literal_keyword() {
        let p = parse_query("\"dog\"");
        assert!(p.semantic.is_empty());
        assert_eq!(p.literals, vec!["dog".to_string()]);
    }

    #[test]
    fn semantic_filtered_by_keyword() {
        // `good "dog"` → semantic "good" filtered by keyword "dog".
        let p = parse_query("good \"dog\"");
        assert_eq!(p.semantic, terms(&[("good", 1.0)]));
        assert_eq!(p.literals, vec!["dog".to_string()]);
    }

    #[test]
    fn multiple_literals_combine() {
        let p = parse_query("\"dog\" \"cat\"");
        assert!(p.semantic.is_empty());
        assert_eq!(p.literals, vec!["dog".to_string(), "cat".to_string()]);
    }

    #[test]
    fn plus_and_minus_terms() {
        assert_eq!(parse_query("dog + cat").semantic, terms(&[("dog", 1.0), ("cat", 1.0)]));
        assert_eq!(parse_query("dog - cat").semantic, terms(&[("dog", 1.0), ("cat", -1.0)]));
    }

    #[test]
    fn multi_word_terms_and_numeric_weights() {
        assert_eq!(
            parse_query("dog is cool - cat").semantic,
            terms(&[("dog is cool", 1.0), ("cat", -1.0)])
        );
        assert_eq!(
            parse_query("dog +1.2 cat").semantic,
            terms(&[("dog", 1.0), ("cat", 1.2)])
        );
        assert_eq!(
            parse_query("+3 dogs are nice -2 cats are mean").semantic,
            terms(&[("dogs are nice", 3.0), ("cats are mean", -2.0)])
        );
    }

    #[test]
    fn mixes_weights_and_literals() {
        let p = parse_query("happy +2 playful \"puppy\"");
        assert_eq!(p.semantic, terms(&[("happy", 1.0), ("playful", 2.0)]));
        assert_eq!(p.literals, vec!["puppy".to_string()]);
    }

    #[test]
    fn empty_query_parses_to_nothing() {
        let p = parse_query("   ");
        assert!(p.semantic.is_empty() && p.literals.is_empty());
    }

    #[test]
    fn normalization_splits_mass_by_ratio() {
        let mut sem = terms(&[("a", 1.0), ("b", 1.0)]);
        let mut prefs: Vec<Preference> = vec![];
        normalize_weights(&mut sem, &mut prefs);
        // Two positive terms share POSITIVE_RATIO evenly.
        let each = POSITIVE_RATIO / 2.0;
        assert!((sem[0].weight - each).abs() < 1e-6);
        assert!((sem[1].weight - each).abs() < 1e-6);
    }

    #[test]
    fn normalization_balances_positive_and_negative() {
        let mut sem = terms(&[("good", 1.0)]);
        let mut prefs = vec![Preference { text: "bad".into(), weight: -1.0 }];
        normalize_weights(&mut sem, &mut prefs);
        assert!((sem[0].weight - POSITIVE_RATIO).abs() < 1e-6);
        assert!((prefs[0].weight + (1.0 - POSITIVE_RATIO)).abs() < 1e-6);
    }
}
