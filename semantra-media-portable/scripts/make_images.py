"""Synthetic still-image fixtures . Usage: make_images.py SRC.jpg OUTDIR"""
import sys, os, struct
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
w, h = im.size  # 1600x972
os.makedirs(out, exist_ok=True)

# EXIF orientations 1..8 on a non-square JPEG (pixels stored unrotated).
for o in range(1, 9):
    ex = Image.Exif()
    ex[0x0112] = o
    im.save(f"{out}/orient{o}.jpg", quality=92, exif=ex.tobytes())

# Alpha: horizontal alpha ramp + a fully transparent hole.
a = Image.linear_gradient("L").rotate(90).resize((w, h))
rgba = im.copy(); rgba.putalpha(a)
for x in range(200, 400):
    for y in range(200, 400):
        rgba.putpixel((x, y), (255, 0, 0, 0))
rgba.save(f"{out}/alpha.png")
rgba.save(f"{out}/alpha.webp", lossless=True)
rgba.save(f"{out}/alpha.tiff")
# Palette PNG with transparency (tRNS).
pal = im.convert("P", palette=Image.ADAPTIVE, colors=200)
pal.info["transparency"] = 0
pal.save(f"{out}/palette_trns.png", transparency=0)
im.convert("P", palette=Image.ADAPTIVE, colors=64).save(f"{out}/palette.png")
# Grayscale variants.
g = im.convert("L")
g.save(f"{out}/gray.jpg", quality=92)
g.save(f"{out}/gray.png")
ga = g.copy(); ga = Image.merge("LA", (g, a)); ga.save(f"{out}/gray_alpha.png")
g.point(lambda v: v * 257).convert("I;16").save(f"{out}/gray16.png")
# WebP lossy, BMP, GIF, TIFF.
im.save(f"{out}/photo.webp", quality=90)
im.save(f"{out}/photo.bmp")
im.convert("P", palette=Image.ADAPTIVE).save(f"{out}/photo.gif")
im.save(f"{out}/photo.tiff", compression="tiff_lzw")
# Multi-frame GIF: first frame must be returned.
frames = [im.convert("P", palette=Image.ADAPTIVE), im.rotate(180).convert("P", palette=Image.ADAPTIVE)]
frames[0].save(f"{out}/anim.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
# Small (never upscaled) and huge (DCT-scaled decode path).
im.resize((300, 182)).save(f"{out}/small.jpg", quality=92)
im.resize((6400, 3888), Image.LANCZOS).save(f"{out}/huge.jpg", quality=90)
im.resize((4000, 2430), Image.LANCZOS).save(f"{out}/large.png")
# Odd aspect for rounding checks.
im.resize((1999, 333)).save(f"{out}/odd.png")
print("ok")
