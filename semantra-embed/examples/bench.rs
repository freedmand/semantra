//! Text throughput sweep: `cargo run --release --example bench -- <model_dir> <corpus.txt>`.
//! Mirrors the lab's mlx-vlm sweep (fixed-length rows, best batch per length).

use std::time::Instant;

use semantra_embed::Model;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = Model::load(&args[1], 768)?;
    let words: Vec<String> = std::fs::read_to_string(&args[2])?.split_whitespace().map(str::to_owned).collect();
    let tok = model.tokenizer();
    let lengths: Vec<i32> = std::env::var("LENS").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![128, 256, 512, 1024, 2048, 4096]);

    for &l in &lengths {
        let mut best = (0.0f64, 0);
        for b in [4, 8, 16, 32, 64, 128].into_iter().filter(|b| (8192..=131072).contains(&(b * l))) {
            // Build `b` rows that tokenize to exactly `l` tokens by decoding a
            // truncated token window back to text.
            let rows: Vec<String> = (0..b)
                .map(|i| {
                    let start = (i as usize * 7919) % (words.len() - 5000);
                    let text = words[start..start + l as usize * 2].join(" ");
                    let ids = tok.encode(text, false).unwrap().get_ids()[..(l as usize - 2)].to_vec();
                    tok.decode(&ids, false).unwrap()
                })
                .collect();
            model.embed_texts(&rows)?; // warmup
            let (mut n, t0) = (0, Instant::now());
            while t0.elapsed().as_secs_f64() < 2.0 {
                model.embed_texts(&rows)?;
                n += 1;
            }
            let tps = (n * b * l) as f64 / t0.elapsed().as_secs_f64();
            if tps > best.0 {
                best = (tps, b);
            }
        }
        println!("L={l:5}: best {:9.0} tok/s at B={}  ({:7.1} chunks/s)", best.0, best.1, best.0 / l as f64);
    }
    Ok(())
}
