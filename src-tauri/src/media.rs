//! The media half of indexing: rendered PDF pages, images, audio and video.
//!
//! Text chunks go through `run_index_pipeline` in `lib.rs`; everything that is
//! embedded from pixels or samples goes through here. A file gets a [`Plan`]
//! (how many units it will produce, so progress covers text + media together),
//! then [`run`] decodes on the calling (blocking) thread while the inference
//! thread embeds: batch k+1 is decoded/rendered while batch k is on the GPU.
//!
//! - **PDF pages** are rendered and embedded as images (alongside the page's
//!   text chunks). Pages with a real text layer use the video-frame budget
//!   (140 soft tokens) — their words are already indexed, the image is for
//!   layout/figures — while near-textless (scanned) pages get the full image
//!   budget (280) so the image itself stays legible.
//! - **Images**: one embedding per file at the full image budget.
//! - **Audio** (files and video soundtracks): 15 s windows every 10 s.
//! - **Video**: frames at 1 fps (stretched for long videos so at most
//!   [`MAX_VIDEO_FRAMES`] are decoded), embedded as 16-frame windows every 8
//!   frames. Each frame goes through the vision tower once; overlapping
//!   windows reuse its soft tokens.

use std::path::Path;

use pdfium_render::prelude::Pdfium;
use semantra_embed::media::{av, image, mel};
use semantra_embed::{EmbedService, Lane};

use crate::extract::{Extracted, FileType};
use crate::pdf;
use crate::store::Modality;

pub const AUDIO_WINDOW_S: f64 = 15.0;
pub const AUDIO_HOP_S: f64 = 10.0;
/// A ragged final audio window is kept only if it adds at least this much
/// audio past the last full window.
const AUDIO_MIN_TAIL_S: f64 = 2.5;
/// Frames per video block; a window spans two consecutive blocks.
pub const VIDEO_BLOCK: usize = 8;
/// Cap on decoded frames per video (the sampling interval grows past ~10 min).
pub const MAX_VIDEO_FRAMES: f64 = 600.0;
/// Page text (chars) at or above which a page counts as text-bearing.
const TEXT_PAGE_CHARS: usize = 200;
/// Pages / audio windows per model call.
const PAGE_BATCH: usize = 8;
const AUDIO_BATCH: usize = 8;
/// Longest side PDF pages and video frames are rasterized/decoded at before the
/// final resize (comfortably above every resize target).
const RENDER_MAX_SIDE: u32 = 1344;

/// What a file's media pass will embed.
pub enum Plan {
    None,
    /// Soft-token budget per page, in page order.
    PdfPages(Vec<u32>),
    Image,
    Audio { windows: usize },
    Video { frame_times: Vec<f64>, interval: f64, duration: f64, audio_windows: usize },
}

/// One embedded media unit, ready to become a store row.
pub struct MediaRow {
    pub modality: Modality,
    pub page: Option<i64>,
    pub time_ms: Option<(i64, i64)>,
    pub vector: Vec<f32>,
}

pub fn plan(filetype: FileType, path: &str, extracted: &Extracted) -> Result<Plan, String> {
    Ok(match filetype {
        FileType::Pdf => Plan::PdfPages(
            extracted
                .segments
                .iter()
                .map(|seg| {
                    let dense = seg.text.chars().filter(|c| !c.is_whitespace()).count() >= TEXT_PAGE_CHARS;
                    if dense { image::FRAME_SOFT_TOKENS } else { image::IMAGE_SOFT_TOKENS }
                })
                .collect(),
        ),
        FileType::Image => Plan::Image,
        FileType::Audio => {
            let p = av::probe(Path::new(path)).map_err(|e| e.to_string())?;
            Plan::Audio { windows: audio_window_count(p.duration_s) }
        }
        FileType::Video => {
            let p = av::probe(Path::new(path)).map_err(|e| e.to_string())?;
            let interval = (p.duration_s / MAX_VIDEO_FRAMES).max(1.0);
            let frame_times = if p.has_video {
                (0..).map(|i| i as f64 * interval).take_while(|&t| t < p.duration_s).collect()
            } else {
                Vec::new()
            };
            let audio_windows = if p.has_audio { audio_window_count(p.duration_s) } else { 0 };
            Plan::Video { frame_times, interval, duration: p.duration_s, audio_windows }
        }
        FileType::Text | FileType::Csv => Plan::None,
    })
}

impl Plan {
    /// Units this plan embeds (pages, images, windows) — for progress.
    pub fn units(&self) -> usize {
        match self {
            Plan::None => 0,
            Plan::PdfPages(p) => p.len(),
            Plan::Image => 1,
            Plan::Audio { windows } => *windows,
            Plan::Video { frame_times, audio_windows, .. } => {
                video_windows(frame_times.len()).len() + audio_windows
            }
        }
    }
}

/// Full windows start every hop while a whole window fits; then one ragged
/// tail window if it adds at least [`AUDIO_MIN_TAIL_S`] (or if the recording
/// is shorter than a window). `run_audio` applies the same rule to the
/// decoded length.
fn audio_window_count(duration: f64) -> usize {
    if duration <= 0.0 {
        return 0;
    }
    let full = full_windows(duration);
    full + usize::from(has_tail(full, duration))
}

fn full_windows(duration: f64) -> usize {
    if duration < AUDIO_WINDOW_S {
        0
    } else {
        1 + ((duration - AUDIO_WINDOW_S) / AUDIO_HOP_S).floor() as usize
    }
}

fn has_tail(full: usize, duration: f64) -> bool {
    let covered = if full == 0 { 0.0 } else { (full - 1) as f64 * AUDIO_HOP_S + AUDIO_WINDOW_S };
    full == 0 || duration - covered >= AUDIO_MIN_TAIL_S
}

/// Video windows as `[first_block, last_block]` over `frames` frames: every
/// pair of consecutive blocks, or the lone block of a short video.
fn video_windows(frames: usize) -> Vec<(usize, usize)> {
    let blocks = frames.div_ceil(VIDEO_BLOCK);
    match blocks {
        0 => vec![],
        1 => vec![(0, 0)],
        n => (0..n - 1).map(|b| (b, b + 1)).collect(),
    }
}

type Pending = futures::channel::oneshot::Receiver<anyhow::Result<Vec<MediaRow>>>;
type Emit<'a> = &'a mut dyn FnMut(Vec<MediaRow>, usize) -> Result<(), String>;

/// Keeps one batch in flight: `push(k + 1)` submits the next batch, then
/// waits for batch k and emits it — so decoding k + 1 overlaps embedding k.
struct Flow<'a> {
    in_flight: Option<(Pending, usize)>,
    emit: Emit<'a>,
}

impl Flow<'_> {
    fn push(&mut self, next: Option<(Pending, usize)>) -> Result<(), String> {
        if let Some((rx, n)) = std::mem::replace(&mut self.in_flight, next) {
            let rows = futures::executor::block_on(rx)
                .map_err(|_| "embedding thread stopped".to_string())?
                .map_err(|e| e.to_string())?;
            (self.emit)(rows, n)?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), String> {
        self.push(None)
    }
}

/// Execute `plan` for the file at `path`, handing each embedded batch to
/// `emit` (with how many units it covers). Blocking: run on a blocking thread.
pub fn run(
    plan: Plan,
    path: &str,
    pdfium: &Pdfium,
    embedder: &EmbedService,
    mut emit: impl FnMut(Vec<MediaRow>, usize) -> Result<(), String>,
) -> Result<(), String> {
    let mut flow = Flow { in_flight: None, emit: &mut emit };
    let err = |e: anyhow::Error| e.to_string();
    match plan {
        Plan::None => {}
        Plan::PdfPages(budgets) => {
            for (b, chunk) in budgets.chunks(PAGE_BATCH).enumerate() {
                let start = b * PAGE_BATCH;
                let indices: Vec<usize> = (start..start + chunk.len()).collect();
                // One open per batch; if that fails, retry page by page and skip
                // only the pages that won't render.
                let renders: Vec<Option<image::Rgb8>> = match pdf::render_pages(pdfium, path, &indices, RENDER_MAX_SIDE) {
                    Ok(r) => r.into_iter().map(Some).collect(),
                    Err(_) => indices
                        .iter()
                        .map(|&i| {
                            pdf::render_page(pdfium, path, i, RENDER_MAX_SIDE)
                                .map_err(|e| eprintln!("[semantra] skipping page {}: {e}", i + 1))
                                .ok()
                        })
                        .collect(),
                };
                let mut prepared = Vec::with_capacity(chunk.len());
                for ((&page, rgb), &budget) in indices.iter().zip(&renders).zip(chunk) {
                    if let Some(rgb) = rgb {
                        prepared.push((page, image::prepare(rgb, budget).map_err(err)?));
                    }
                }
                let rx = embedder.submit(Lane::Background, move |m| {
                    embed_images(m, prepared, |page| MediaRow {
                        modality: Modality::Image,
                        page: Some(page as i64),
                        time_ms: None,
                        vector: Vec::new(),
                    })
                });
                flow.push(Some((rx, chunk.len())))?;
            }
        }
        Plan::Image => {
            let rgb = image::decode(Path::new(path)).map_err(err)?;
            let prepared = vec![(0, image::prepare(&rgb, image::IMAGE_SOFT_TOKENS).map_err(err)?)];
            let rx = embedder.submit(Lane::Background, move |m| {
                embed_images(m, prepared, |_| MediaRow {
                    modality: Modality::Image,
                    page: None,
                    time_ms: None,
                    vector: Vec::new(),
                })
            });
            flow.push(Some((rx, 1)))?;
        }
        Plan::Audio { .. } => run_audio(path, embedder, &mut flow)?,
        Plan::Video { frame_times, interval, duration, audio_windows } => {
            run_video(path, embedder, &frame_times, interval, duration, &mut flow)?;
            if audio_windows > 0 {
                run_audio(path, embedder, &mut flow)?;
            }
        }
    }
    flow.drain()
}

/// Embed prepared images on the inference thread, batching same-grid ones.
/// `row` builds each row's metadata from its key; the vector is filled in.
fn embed_images(
    m: &semantra_embed::Model,
    prepared: Vec<(usize, image::Prepared)>,
    row: impl Fn(usize) -> MediaRow,
) -> anyhow::Result<Vec<MediaRow>> {
    let mut out = Vec::with_capacity(prepared.len());
    let mut grids: Vec<(u32, u32)> = prepared.iter().map(|(_, p)| p.grid()).collect();
    grids.sort();
    grids.dedup();
    for g in grids {
        let group: Vec<&(usize, image::Prepared)> = prepared.iter().filter(|(_, p)| p.grid() == g).collect();
        let grid = image::stack(&group.iter().map(|(_, p)| p).collect::<Vec<_>>())?;
        let emb = m.embed_visual(&grid, 1)?;
        for ((key, _), vector) in group.iter().zip(emb.rows) {
            out.push(MediaRow { vector, ..row(*key) });
        }
    }
    Ok(out)
}

/// Submit equal-length audio clips `(start_sample, samples)` as one batch.
fn submit_audio(embedder: &EmbedService, flow: &mut Flow, clips: Vec<(usize, Vec<f32>)>) -> Result<(), String> {
    if clips.is_empty() {
        return Ok(());
    }
    let n = clips.len();
    let rate = mel::SAMPLE_RATE as f64;
    let rx = embedder.submit(Lane::Background, move |m| {
        let refs: Vec<&[f32]> = clips.iter().map(|(_, c)| c.as_slice()).collect();
        let emb = m.embed_audio(&refs)?;
        let ms = |s: usize| (s as f64 * 1000.0 / rate) as i64;
        Ok(clips
            .iter()
            .zip(emb.rows)
            .map(|((start, clip), vector)| MediaRow {
                modality: Modality::Audio,
                page: None,
                time_ms: Some((ms(*start), ms(start + clip.len()))),
                vector,
            })
            .collect())
    });
    flow.push(Some((rx, n)))
}

/// Stream the first audio track and embed it in overlapping windows. Only a
/// window's worth of samples (plus one decoder chunk) is buffered at a time.
fn run_audio(path: &str, embedder: &EmbedService, flow: &mut Flow) -> Result<(), String> {
    let rate = mel::SAMPLE_RATE as f64;
    let (win, hop) = ((AUDIO_WINDOW_S * rate) as usize, (AUDIO_HOP_S * rate) as usize);
    let mut buf: Vec<f32> = Vec::new();
    let mut buf_start = 0usize; // absolute sample index of buf[0]
    let mut next = 0usize; // absolute start of the next window
    let mut batch: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut failure: Option<String> = None;

    av::stream_audio(Path::new(path), |chunk| {
        buf.extend_from_slice(chunk);
        while buf_start + buf.len() >= next + win {
            let off = next - buf_start;
            batch.push((next, buf[off..off + win].to_vec()));
            next += hop;
            let consumed = (next - buf_start).min(buf.len());
            buf.drain(..consumed);
            buf_start += consumed;
            if batch.len() == AUDIO_BATCH {
                if let Err(e) = submit_audio(embedder, flow, std::mem::take(&mut batch)) {
                    failure = Some(e.clone());
                    anyhow::bail!(e);
                }
            }
        }
        Ok(())
    })
    .map_err(|e| failure.take().unwrap_or_else(|| e.to_string()))?;
    submit_audio(embedder, flow, std::mem::take(&mut batch))?;

    // Ragged tail (same rule as `audio_window_count`), embedded alone.
    let total = buf_start + buf.len();
    let full = next / hop;
    if next < total && has_tail(full, total as f64 / rate) {
        let tail = buf[next - buf_start..].to_vec();
        if tail.len() >= mel::SAMPLE_RATE as usize / 4 {
            submit_audio(embedder, flow, vec![(next, tail)])?;
        }
    }
    Ok(())
}

/// Decode frames block by block and embed each window of two blocks, reusing
/// the previous block's soft tokens.
///
/// Pipelined like the other passes: block k + 1 is decoded on this thread
/// while the GPU encodes block k. That is safe because background jobs run in
/// FIFO order on the single inference thread, so each job can take the
/// previous block's soft tokens from `slot` and leave its own for the next.
fn run_video(
    path: &str,
    embedder: &EmbedService,
    times: &[f64],
    interval: f64,
    duration: f64,
    flow: &mut Flow,
) -> Result<(), String> {
    let blocks: Vec<&[f64]> = times.chunks(VIDEO_BLOCK).collect();
    let slot: std::sync::Arc<std::sync::Mutex<Option<semantra_embed::SoftFrames>>> = Default::default();
    let lone = blocks.len() == 1;
    for (b, block) in blocks.iter().enumerate() {
        let frames = av::frames_at(Path::new(path), block, RENDER_MAX_SIDE, interval / 2.0).map_err(|e| e.to_string())?;
        let prepared: Vec<image::Prepared> = frames
            .iter()
            .map(|f| image::prepare(f, image::FRAME_SOFT_TOKENS))
            .collect::<anyhow::Result<_>>()
            .map_err(|e| e.to_string())?;
        // This block closes the window that started at the previous block (or,
        // for a one-block video, is the whole window).
        let start = if b == 0 { block[0] } else { blocks[b - 1][0] };
        let end = (block[block.len() - 1] + interval).min(duration);
        let emits = b > 0 || lone;
        let slot = std::sync::Arc::clone(&slot);
        let rx = embedder.submit(Lane::Background, move |m| {
            let grid = image::stack(&prepared.iter().collect::<Vec<_>>())?;
            let cur = m.visual_soft_tokens(&grid)?;
            let prev = slot.lock().unwrap().take();
            let vector = match (&prev, lone) {
                (Some(p), _) => Some(m.embed_frame_window(&[p, &cur])?),
                (None, true) => Some(m.embed_frame_window(&[&cur])?),
                (None, false) => None, // first of several: wait for its pair
            };
            *slot.lock().unwrap() = Some(cur);
            let ms = |s: f64| (s * 1000.0) as i64;
            Ok(vector
                .map(|vector| MediaRow {
                    modality: Modality::Video,
                    page: None,
                    time_ms: Some((ms(start), ms(end))),
                    vector,
                })
                .into_iter()
                .collect())
        });
        flow.push(Some((rx, usize::from(emits))))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_windows_cover_the_recording() {
        assert_eq!(audio_window_count(0.0), 0);
        assert_eq!(audio_window_count(8.0), 1); // shorter than a window
        assert_eq!(audio_window_count(15.0), 1);
        assert_eq!(audio_window_count(16.0), 1); // 1 s past the window: no tail
        assert_eq!(audio_window_count(18.0), 2); // 3 s past: tail
        assert_eq!(audio_window_count(60.0), 6); // 0,10,20,30,40 (to 55) + tail 50–60
    }

    #[test]
    fn video_windows_pair_consecutive_blocks() {
        assert!(video_windows(0).is_empty());
        assert_eq!(video_windows(5), vec![(0, 0)]);
        assert_eq!(video_windows(8), vec![(0, 0)]);
        assert_eq!(video_windows(9), vec![(0, 1)]);
        assert_eq!(video_windows(30), vec![(0, 1), (1, 2), (2, 3)]);
    }
}
