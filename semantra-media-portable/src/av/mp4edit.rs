//! ISO-BMFF (MP4/M4A/MOV) edit lists for audio tracks.
//!
//! AAC streams start with encoder priming (typically 1024–2112 samples) that
//! the `elst` box tells players to skip, and end with padding that its segment
//! duration cuts off. AVFoundation honors this; Symphonia 0.5 parses `elst`
//! but ignores it, which leaves its output ~23–64 ms late and slightly long.
//! This reads just enough of the box tree (seeking past `mdat` and sample
//! tables) to apply the same trim ourselves.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Trim for one track, in samples at the track's sample rate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Edit {
    /// Silence to insert first (leading empty edits).
    pub lead: u64,
    /// Decoded samples to drop (priming).
    pub skip: u64,
    /// Samples to keep after the skip; `None` = to the end.
    pub keep: Option<u64>,
}

struct Boxes<'a> {
    f: &'a mut File,
    pos: u64,
    end: u64,
}

impl Iterator for Boxes<'_> {
    /// (fourcc, payload start, payload end)
    type Item = ([u8; 4], u64, u64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 8 > self.end {
            return None;
        }
        self.f.seek(SeekFrom::Start(self.pos)).ok()?;
        let mut h = [0u8; 8];
        self.f.read_exact(&mut h).ok()?;
        let size32 = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as u64;
        let kind = [h[4], h[5], h[6], h[7]];
        let (start, size) = match size32 {
            1 => {
                let mut l = [0u8; 8];
                self.f.read_exact(&mut l).ok()?;
                (self.pos + 16, u64::from_be_bytes(l))
            }
            0 => (self.pos + 8, self.end - self.pos),
            s => (self.pos + 8, s),
        };
        let end = self.pos.checked_add(size)?.min(self.end);
        if size < 8 || start > end {
            return None;
        }
        self.pos = end;
        Some((kind, start, end))
    }
}

fn children(f: &mut File, start: u64, end: u64) -> Vec<([u8; 4], u64, u64)> {
    Boxes { f, pos: start, end }.collect()
}

fn read_at(f: &mut File, at: u64, len: u64) -> Option<Vec<u8>> {
    if len > 16 << 20 {
        return None;
    }
    f.seek(SeekFrom::Start(at)).ok()?;
    let mut v = vec![0u8; len as usize];
    f.read_exact(&mut v).ok()?;
    Some(v)
}

fn be32(b: &[u8], at: usize) -> Option<u64> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?) as u64)
}

fn be64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// Timescale from an `mvhd` or `mdhd` payload.
fn timescale(b: &[u8]) -> Option<u64> {
    if b.first()? == &1 {
        be32(b, 20)
    } else {
        be32(b, 12)
    }
}

/// `elst` entries as (segment_duration, media_time).
fn elst(b: &[u8]) -> Option<Vec<(u64, i64)>> {
    let v1 = b.first()? == &1;
    let n = be32(b, 4)? as usize;
    let mut out = Vec::with_capacity(n.min(64));
    let mut at = 8;
    for _ in 0..n.min(64) {
        if v1 {
            out.push((be64(b, at)?, be64(b, at + 8)? as i64));
            at += 20;
        } else {
            out.push((be32(b, at)?, be32(b, at + 4)? as u32 as i32 as i64));
            at += 12;
        }
    }
    Some(out)
}

/// The edit of the `trak_index`-th track (0-based, file order — Symphonia's
/// isomp4 track ids) converted to samples at `rate`. `None` if the file isn't
/// ISO-BMFF or the track has no edit list.
pub fn audio_edit(path: &Path, trak_index: usize, rate: u32) -> Option<Edit> {
    let mut f = File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let top = children(&mut f, 0, len);
    // Only real ISO-BMFF: the first box is one of the usual top-level ones.
    if !matches!(&top.first()?.0, b"ftyp" | b"moov" | b"free" | b"wide" | b"skip" | b"mdat") {
        return None;
    }
    let (_, ms, me) = top.into_iter().find(|b| &b.0 == b"moov")?;
    let moov = children(&mut f, ms, me);
    let (_, hs, he) = *moov.iter().find(|b| &b.0 == b"mvhd")?;
    let movie_ts = timescale(&read_at(&mut f, hs, he - hs)?)?;
    let (_, ts, te) = *moov.iter().filter(|b| &b.0 == b"trak").nth(trak_index)?;
    let trak = children(&mut f, ts, te);
    let (_, es, ee) = *trak.iter().find(|b| &b.0 == b"edts")?;
    let (_, ls, le) = children(&mut f, es, ee).into_iter().find(|b| &b.0 == b"elst")?;
    let entries = elst(&read_at(&mut f, ls, le - ls)?)?;
    let (_, ds, de) = *trak.iter().find(|b| &b.0 == b"mdia")?;
    let (_, hs, he) = children(&mut f, ds, de).into_iter().find(|b| &b.0 == b"mdhd")?;
    let media_ts = timescale(&read_at(&mut f, hs, he - hs)?)?;
    if movie_ts == 0 || media_ts == 0 {
        return None;
    }
    let at_rate = |v: u64, scale: u64| ((v as u128 * rate as u128 + scale as u128 / 2) / scale as u128) as u64;
    let mut edit = Edit::default();
    for (dur, media_time) in entries {
        if media_time == -1 {
            edit.lead += at_rate(dur, movie_ts);
            continue;
        }
        edit.skip = at_rate(media_time.max(0) as u64, media_ts);
        edit.keep = (dur > 0).then(|| at_rate(dur, movie_ts));
        break; // later edits (rare in audio) are not applied
    }
    (edit != Edit::default()).then_some(edit)
}

/// Applies an [`Edit`] to a stream of samples.
pub struct Trim {
    lead: u64,
    skip: u64,
    left: Option<u64>,
}

impl Trim {
    pub fn new(e: Edit) -> Self {
        Self { lead: e.lead, skip: e.skip, left: e.keep }
    }

    /// Silence to emit before the first samples (once).
    pub fn take_lead(&mut self) -> u64 {
        std::mem::take(&mut self.lead)
    }

    /// The part of `s` that survives the edit.
    pub fn apply<'s>(&mut self, mut s: &'s [f32]) -> &'s [f32] {
        let k = (self.skip.min(s.len() as u64)) as usize;
        self.skip -= k as u64;
        s = &s[k..];
        if let Some(left) = &mut self.left {
            let n = (*left).min(s.len() as u64) as usize;
            *left -= n as u64;
            s = &s[..n];
        }
        s
    }
}
