//! Phase timing (tokenize vs forward) for one batch shape.
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = semantra_embed::Model::load(&a[1], 768)?;
    let words: Vec<&str> = Box::leak(std::fs::read_to_string(&a[2])?.into_boxed_str()).split_whitespace().collect();
    for (l, b) in [(128usize, 64usize), (512, 16), (1024, 8)] {
        let rows: Vec<String> = (0..b).map(|i| words[i * 1500..i * 1500 + (l as f32 * 0.72) as usize].join(" ")).collect();
        model.embed_texts(&rows)?;
        let t = Instant::now();
        for _ in 0..5 { model.tokenizer().encode_batch(rows.clone(), true).unwrap(); }
        let tok_ms = t.elapsed().as_secs_f64() * 200.0;
        let t = Instant::now();
        for _ in 0..5 { model.embed_texts(&rows)?; }
        let all_ms = t.elapsed().as_secs_f64() * 200.0;
        let n = model.tokenizer().encode_batch(rows.clone(), true).unwrap()[0].len();
        println!("L~{n:5} B={b:3}: tokenize {tok_ms:7.1} ms   total {all_ms:7.1} ms   -> {:6.0} tok/s excluding tokenize", (n * b) as f64 / ((all_ms - tok_ms) / 1000.0));
    }
    Ok(())
}
