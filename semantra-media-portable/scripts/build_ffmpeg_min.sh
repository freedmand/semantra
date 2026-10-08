#!/bin/bash
# Minimal decode-only LGPL ffmpeg CLI for the Semantra sidecar.
#   build_ffmpeg_min.sh <ffmpeg-source-dir> <build-dir> [extra configure args]
# Linux/Windows release builds should add --enable-libdav1d (BSD, AV1) and
# cross-compile flags (e.g. --target-os=mingw32 --arch=x86_64
# --cross-prefix=x86_64-w64-mingw32-), and keep --disable-gpl (the default)
# and no --enable-nonfree.
set -euo pipefail
SRC=$(cd "$1" && pwd); OUT=$2; shift 2
mkdir -p "$OUT" && cd "$OUT"
"$SRC/configure" \
  --prefix="$OUT/install" \
  --disable-everything --disable-autodetect --disable-doc --disable-debug --disable-network \
  --disable-ffplay --disable-ffprobe --enable-ffmpeg --enable-static --disable-shared \
  --enable-zlib \
  --enable-protocol=file,pipe \
  --enable-demuxer=mov,matroska,avi,mpegts,mpegps,flv,ogg,wav,w64,aiff,caf,mp3,aac,ac3,eac3,flac,asf,m4v,h264,hevc,ivf,mxf,amr,loas \
  --enable-parser=h264,hevc,vp8,vp9,av1,mpeg4video,mpegvideo,mpegaudio,aac,aac_latm,ac3,opus,vorbis,flac,mjpeg,dca,vc1,h263 \
  --enable-decoder=h264,hevc,vp8,vp9,mpeg4,msmpeg4v1,msmpeg4v2,msmpeg4v3,mpeg1video,mpeg2video,mjpeg,prores,dnxhd,theora,wmv1,wmv2,wmv3,vc1,h263,flv,png,rawvideo \
  --enable-decoder=aac,aac_latm,mp1,mp2,mp3,mp3float,opus,vorbis,flac,alac,ac3,eac3,dca,truehd,mlp,wmav1,wmav2,wmapro,amrnb,amrwb,'pcm*','adpcm*' \
  --enable-encoder=ppm,pcm_f32le \
  --enable-muxer=image2pipe,pcm_f32le,null \
  --enable-filter=scale,select,showinfo,format,aformat,aresample,null,anull,transpose,hflip,vflip,rotate,setpts,copy,xstack,crop \
  "$@"
mkdir -p fftools/resources  # ffmpeg 9 out-of-tree builds expect it
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" ffmpeg
strip ffmpeg 2>/dev/null || true
ls -la ffmpeg
