//! Vision throughput: `cargo run --release --example bench_vision -- <model_dir> <image>`.
use std::time::Instant;
use semantra_embed::{media::image, Model};
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = Model::load(&a[1], 768)?;
    let rgb = image::decode(std::path::Path::new(&a[2]))?;
    for tokens in [image::IMAGE_SOFT_TOKENS, image::FRAME_SOFT_TOKENS, 70] {
        let p = image::prepare(&rgb, tokens)?;
        for b in [1, 4, 8, 16] {
            let batch: Vec<&image::Prepared> = std::iter::repeat(&p).take(b).collect();
            let grid = image::stack(&batch)?;
            model.embed_visual(&grid, 1)?;
            let (t, mut n) = (Instant::now(), 0);
            while t.elapsed().as_secs_f64() < 2.0 { model.embed_visual(&grid, 1)?; n += b; }
            println!("{tokens:4} soft tokens, grid {:?}, B={b:2}: {:6.1} images/s", p.grid(), n as f64 / t.elapsed().as_secs_f64());
        }
    }
    Ok(())
}
