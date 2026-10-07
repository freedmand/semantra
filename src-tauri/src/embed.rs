//! Reusable, Tauri-agnostic embedding helper: batch planning.
//!
//! Inference itself runs on the `semantra_embed::EmbedService` thread; this
//! module decides *how* a file's text chunks are grouped into model calls. It
//! deliberately does NOT depend on Tauri, so it can be unit-tested freely.

/// Token budget per model call (rows × padded length). Measured on M5: the
/// encoder's throughput plateaus around 8–16k tokens per pass at our chunk
/// lengths, and activation memory stays well under 1 GB.
pub const BATCH_TOKENS: usize = 16_384;

/// Row cap per call, so a file of tiny chunks (CSV cells) still streams
/// progress and yields to interactive queries regularly.
pub const MAX_BATCH_ROWS: usize = 256;

/// Group chunk indices into model calls: sort by token length (so rows in a
/// batch pad to nearly the same length) and greedily fill each batch while
/// `rows × longest_row <= budget`. Every index appears exactly once.
pub fn plan_batches(token_lens: &[usize], budget: usize, max_rows: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..token_lens.len()).collect();
    order.sort_by_key(|&i| token_lens[i]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in order {
        // Sorted ascending, so the newcomer is the batch's longest row.
        let longest = token_lens[i].max(1);
        if !cur.is_empty() && ((cur.len() + 1) * longest > budget || cur.len() == max_rows) {
            batches.push(std::mem::take(&mut cur));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        batches.push(cur);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_once_and_within_budget() {
        let lens = vec![5, 300, 12, 265, 265, 1, 40, 265, 700, 3];
        let batches = plan_batches(&lens, 600, 4);
        let mut seen: Vec<usize> = batches.iter().flatten().copied().collect();
        seen.sort();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>());
        for b in &batches {
            let longest = b.iter().map(|&i| lens[i]).max().unwrap();
            // A single oversized row may exceed the budget alone, never in company.
            assert!(b.len() == 1 || b.len() * longest <= 600, "{b:?}");
            assert!(b.len() <= 4);
        }
    }

    #[test]
    fn rows_are_grouped_by_similar_length() {
        let lens = vec![256, 8, 256, 8, 256, 8];
        let batches = plan_batches(&lens, 10_000, 3);
        assert_eq!(batches, vec![vec![1, 3, 5], vec![0, 2, 4]]);
    }

    #[test]
    fn empty_input_plans_nothing() {
        assert!(plan_batches(&[], 100, 10).is_empty());
    }
}
