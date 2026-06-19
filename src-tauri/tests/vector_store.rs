//! Integration tests for the LanceDB-backed `VectorStore`.
//!
//! These use **synthetic** unit-norm vectors (no embedding model) so they are
//! fast and deterministic, and they double as the proof that LanceDB builds and
//! links cleanly on this target. They exercise file-scoped insertion + search,
//! the IVF-HNSW-SQ ANN index, the BM25 full-text index, and per-file deletion.
//!
//! Run with: cargo test --release

use semantra_lib::store::{ChunkRow, SearchMode, VectorStore};

const DIM: usize = 16;
const FILE: &str = "fileA";

/// Tiny deterministic PRNG (no `rand` dependency) → values in roughly [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn unit_vec(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        normalize(v)
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn basis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i] = 1.0;
    v
}

fn row(file: &str, text: &str, vector: Vec<f32>) -> ChunkRow {
    ChunkRow {
        sha512: file.to_string(),
        text: text.to_string(),
        char_start: 0,
        char_end: text.chars().count() as i64,
        page: None,
        page_char_start: 0,
        pipeline_version: "test".to_string(),
        vector,
    }
}

async fn temp_store() -> (VectorStore, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let conn = lancedb::connect(tmp.path().to_str().unwrap())
        .execute()
        .await
        .unwrap();
    (VectorStore::open(conn, DIM as i32).await.unwrap(), tmp)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_search_finds_planted() {
    let (mut store, _t) = temp_store().await;

    let rows: Vec<ChunkRow> = (0..5).map(|i| row(FILE, &format!("doc{i}"), basis(i))).collect();
    store.insert(&rows).await.unwrap();
    assert_eq!(store.count_for(&[FILE.into()]).await.unwrap(), 5);

    // Query close to e_2 (basis 2 plus a little noise on another axis).
    let mut q = basis(2);
    q[0] = 0.1;
    let q = normalize(q);

    let hits = store
        .search(&q, 3, SearchMode::Exact, &[FILE.into()])
        .await
        .unwrap();
    assert_eq!(hits.len(), 3, "should return the requested k");
    assert_eq!(hits[0].text, "doc2", "nearest neighbour must be the planted doc");
    assert!(hits[0].score > 0.9, "score should be high, got {}", hits[0].score);
    assert!(hits[0].distance <= hits[1].distance);
    assert!(hits[1].distance <= hits[2].distance);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ann_index_builds_and_recalls() {
    let (mut store, _t) = temp_store().await;

    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let n = 300usize;
    let mut rows: Vec<ChunkRow> = (0..n)
        .map(|i| row(FILE, &format!("rand{i}"), rng.unit_vec(DIM)))
        .collect();

    let needle = normalize(basis(0).iter().zip(basis(7)).map(|(a, b)| a + b).collect());
    rows.push(row(FILE, "needle", needle.clone()));

    store.insert(&rows).await.unwrap();
    assert_eq!(store.count_for(&[FILE.into()]).await.unwrap(), n + 1);

    store.maybe_build_ann().await.unwrap();

    let ann = store
        .search(&needle, 5, SearchMode::Ann, &[FILE.into()])
        .await
        .unwrap();
    assert_eq!(ann[0].text, "needle", "ANN should recall the planted needle first");
    assert!(ann[0].score > 0.99, "needle score should be ~1, got {}", ann[0].score);

    let exact = store
        .search(&needle, 5, SearchMode::Exact, &[FILE.into()])
        .await
        .unwrap();
    assert_eq!(exact[0].text, "needle");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_text_search_matches_keyword() {
    let (mut store, _t) = temp_store().await;
    store
        .insert(&[
            row(FILE, "the quick brown fox", basis(0)),
            row(FILE, "a lazy sleeping dog", basis(1)),
            row(FILE, "another brown bear", basis(2)),
        ])
        .await
        .unwrap();
    store.ensure_fts_index().await.unwrap();

    let hits = store.fts_search("brown", 10, &[FILE.into()]).await.unwrap();
    let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
    assert!(texts.iter().all(|t| t.contains("brown")), "got {texts:?}");
    assert_eq!(hits.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_file_clears_only_that_file() {
    let (mut store, _t) = temp_store().await;
    store
        .insert(&[row("a", "x", basis(0)), row("b", "y", basis(1))])
        .await
        .unwrap();
    assert_eq!(store.count_for(&["a".into(), "b".into()]).await.unwrap(), 2);

    store.delete_file("a").await.unwrap();
    assert_eq!(store.count_for(&["a".into()]).await.unwrap(), 0);
    assert_eq!(store.count_for(&["b".into()]).await.unwrap(), 1);

    // Empty file filter → no hits, not an error.
    let hits = store
        .search(&basis(0), 5, SearchMode::Exact, &[])
        .await
        .unwrap();
    assert!(hits.is_empty());
}
