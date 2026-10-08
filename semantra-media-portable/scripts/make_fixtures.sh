#!/bin/bash
# Build all test fixtures into testdata/ (needs ffmpeg, sips, python3+PIL).
set -euo pipefail
cd "$(dirname "$0")/.."
M=~/scraps/eg2-lab/media
APP="$HOME/Library/Application Support/com.semantra.app/files"
I=testdata/img; A=testdata/aud; V=testdata/vid
mkdir -p $I $A $V testdata/app
python3 scripts/make_images.py $M/the_beach.jpg $I
ff() { ffmpeg -hide_banner -loglevel error -y "$@"; }
# 16-bit RGB / RGBA PNG and 16-bit TIFF.
ff -i $M/the_beach.jpg -pix_fmt rgb48be $I/rgb16.png
ff -i $I/alpha.png -pix_fmt rgba64be $I/rgba16.png
ff -i $M/the_beach.jpg -pix_fmt rgb48le $I/rgb16.tiff
# CMYK JPEG and wide-gamut (Display P3 / Adobe-ish) JPEGs with embedded ICC.
sips -s format jpeg -m "/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc" $M/big_sur_road.jpg --out $I/cmyk.jpg >/dev/null
sips -m "/System/Library/ColorSync/Profiles/Display P3.icc" $M/big_sur_coastline.jpg --out $I/p3.jpg >/dev/null
sips -s format png -m "/System/Library/ColorSync/Profiles/Display P3.icc" $M/tree.jpg --out $I/p3.png >/dev/null
# HEIC (sips writes tile grids, `irot` for EXIF orientation, ICC `colr`).
sips -s format heic $M/the_beach.jpg --out $I/beach.heic >/dev/null
sips -s format heic $I/orient6.jpg --out $I/orient6.heic >/dev/null
sips -s format heic $I/p3.jpg --out $I/p3.heic >/dev/null
sips -s format heic $I/huge.jpg --out $I/huge.heic >/dev/null
# App files (read-only source): copy the small ones.
for f in "$APP"/*.jpg "$APP"/*.wav "$APP"/*.aiff; do cp -n "$f" testdata/app/ || true; done
cp -n "$APP"/b809*.mp4 testdata/app/ || true

# Audio: stereo 44.1/48 kHz (L = pasta, R = stocks) and every codec.
ff -i $M/pasta.wav -i $M/stocks.wav -filter_complex "[0][1]amerge=inputs=2,aresample=44100" -ac 2 -c:a pcm_s16le $A/stereo44.wav
ff -i $A/stereo44.wav -ar 48000 -c:a pcm_s24le $A/stereo48_s24.wav
ff -i $A/stereo44.wav -c:a pcm_f32le $A/stereo44_f32.wav
ff -i $A/stereo44.wav -c:a pcm_s16be $A/stereo44.aiff
ff -i $A/stereo44.wav -c:a libmp3lame -b:a 192k $A/stereo44.mp3
ff -i $A/stereo44.wav -c:a libmp3lame -q:a 4 -write_xing 0 $A/noxing.mp3
ff -i $A/stereo44.wav -c:a aac -b:a 160k $A/stereo44.m4a
ff -i $A/stereo44.wav -c:a alac $A/stereo44_alac.m4a
ff -i $A/stereo44.wav -c:a flac $A/stereo44.flac
ff -i $A/stereo44.wav -c:a vorbis -strict -2 $A/stereo44.ogg
ff -i $A/stereo48_s24.wav -c:a libopus -b:a 96k $A/stereo48.opus
ff -i $A/stereo48_s24.wav -c:a libopus -b:a 96k $A/stereo48_opus.webm
ff -i $A/stereo44.wav -c:a aac_at -profile:a 4 -b:a 48k $A/he_aac.m4a || true
ff -i $A/stereo44.wav -ar 8000 -ac 1 -c:a pcm_u8 $A/mono8k_u8.wav
ff -i $A/stereo48_s24.wav -filter_complex "[0]pan=5.1|FL=c0|FR=c1|FC=0.5*c0+0.5*c1|LFE=0*c0|BL=c0|BR=c1" -c:a pcm_s16le $A/surround51.wav
# MP3 with cover art (an attached pic is not video).
ff -i $A/stereo44.wav -i $M/tree.jpg -map 0:a -map 1:v -c:a libmp3lame -c:v mjpeg -disposition:v attached_pic $A/cover.mp3
# Long audio for throughput: the 37-min video's soundtrack as AAC m4a and MP3.
ff -i "$APP"/4a9e*.mp4 -vn -c:a copy $A/long37.m4a
ff -i $A/long37.m4a -c:a libmp3lame -b:a 128k $A/long37.mp3

# Video: rotated, HEVC, VP9/WebM, AV1/MKV, MOV, odd size, from clip.mp4.
C=$M/clip.mp4
ff -display_rotation:v:0 90 -i $C -c copy $V/rot90.mp4
ff -display_rotation:v:0 -90 -i $C -c copy $V/rot270.mp4
ff -display_rotation:v:0 180 -i $C -c copy $V/rot180.mp4
ff -i $C -c:v libx265 -tag:v hvc1 -crf 24 -preset fast -c:a copy $V/hevc.mp4
ff -i $C -c:v libx264 -crf 20 -g 48 -c:a copy $V/h264.mov
ff -i $C -c:v libvpx-vp9 -crf 32 -b:v 0 -deadline realtime -cpu-used 8 -c:a libopus $V/vp9.webm
ff -i $C -c:v libsvtav1 -crf 35 -preset 10 -c:a libopus $V/av1.mkv
ff -i $C -vf scale=1918:1078,setsar=1 -c:v libx264 -crf 20 -an $V/odd_noaudio.mp4
ff -i $C -vf scale=640:-2 -c:v libx264 -crf 20 -an $V/small.mp4
ff -i $C -vf "scale=3840:2160" -c:v libx264 -crf 22 -preset veryfast -an -t 5 $V/uhd.mp4
ff -ss 25 -t 10 -i "$APP"/165d*.mp4 -vf scale=640:360 -c:v libx264 -crf 18 -an $V/untagged_sd.mp4
ff -ss 25 -t 10 -i "$APP"/165d*.mp4 -c:v copy -an $V/untagged_hd.mp4

# Color-tag matrix (parity `color`): the same 3 s of untagged 1080p content
# re-tagged every way that changes AVFoundation's color handling.
CO=testdata/color; mkdir -p $CO
enc() { local name=$1 vf=$2; shift 2; ff -ss 30 -t 3 -i "$APP"/165d*.mp4 -vf "scale=1280:720$vf" "$@" -c:v libx264 -crf 16 -an $CO/$name.mp4; }
enc untagged ""
enc m601only ",setparams=colorspace=smpte170m"
enc m470bgonly ",setparams=colorspace=bt470bg"
enc m709only ",setparams=colorspace=bt709"
enc full709 ",setparams=colorspace=bt709:color_trc=bt709:color_primaries=bt709"
enc full601 ",setparams=colorspace=smpte170m:color_trc=smpte170m:color_primaries=smpte170m"
enc trc709only ",setparams=color_trc=bt709"
enc prim709only ",setparams=color_primaries=bt709"
enc srgbtrc ",setparams=colorspace=bt709:color_trc=iec61966-2-1:color_primaries=bt709"
enc linear ",setparams=colorspace=bt709:color_trc=linear:color_primaries=bt709"
enc jrange ",format=yuvj420p"
enc jrange470 ",format=yuvj420p,setparams=colorspace=bt470bg"
hdr() { local name=$1 trc=$2; ff -ss 30 -t 3 -i "$APP"/165d*.mp4 -vf "scale=1280:720,format=yuv420p10le,setparams=colorspace=bt2020nc:color_trc=$trc:color_primaries=bt2020" -c:v libx265 -tag:v hvc1 -crf 18 -preset fast -x265-params "colorprim=bt2020:transfer=$trc:colormatrix=bt2020nc:log-level=error" -an $CO/$name.mp4; }
hdr hlg10 arib-std-b67
hdr pq10 smpte2084
ff -ss 30 -t 3 -i "$APP"/165d*.mp4 -vf "scale=1280:720,format=yuv420p10le,setparams=colorspace=bt709:color_trc=bt709:color_primaries=bt709" -c:v libx265 -tag:v hvc1 -crf 18 -preset fast -x265-params log-level=error -an $CO/sdr10.mp4

# Resampler probes: tones (parity `phase`, `response`) and one-channel
# surround files (parity `downmix`).
mkdir -p testdata/tones testdata/resp testdata/downmix
for r in 8000 11025 22050 24000 32000 44100 48000 88200 96000; do
  ff -f lavfi -i "sine=f=1000:d=3:sample_rate=$r" -c:a pcm_f32le testdata/tones/t$r.wav
done
for f in 1000 4000 6000 7000 7400 7600 7800 7900 8100 8500 9000; do
  ff -f lavfi -i "sine=f=$f:d=2:sample_rate=44100" -c:a pcm_f32le testdata/resp/r44_$f.wav
  ff -f lavfi -i "sine=f=$f:d=2:sample_rate=48000" -c:a pcm_f32le testdata/resp/r48_$f.wav
done
for lay in mono stereo 3.0 quad 5.0 5.1 7.1; do
  n=$(ffmpeg -hide_banner -layouts 2>/dev/null | awk -v l="$lay" '$1==l{print $2}' | tr '+' '\n' | wc -l | tr -d ' ')
  for ((k=0; k<n; k++)); do
    ff -f lavfi -i "sine=f=440:d=2:sample_rate=48000" -af "pan=$lay|c$k=c0" -c:a pcm_f32le testdata/downmix/${lay}_c$k.wav
  done
done
echo fixtures ok
