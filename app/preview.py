"""Watermarked preview JPEGs for unpaid result pages.

Watermark sits on the white background only; the subject stays clean on top.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

log = logging.getLogger("gosphoto-gate")

PREVIEW_MAX_SIDE = 480
WATERMARK_TEXT = "Госфото"
WATERMARK_FILL = (25, 25, 25, 165)
WATERMARK_HALO = (255, 255, 255, 90)
# Pixels farther than this from pure white count as subject (same idea as readiness).
SUBJECT_WHITE_TOL = 40


def make_preview_jpeg(source_jpeg: bytes, *, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    """Downscale + background-only watermark. Falls back to original bytes on error."""
    if not source_jpeg:
        return b""
    try:
        img = Image.open(io.BytesIO(source_jpeg)).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        watermarked_bg = _watermarked_white(img.size)
        subject = img.convert("RGBA")
        subject.putalpha(_subject_alpha(img))
        composed = Image.alpha_composite(watermarked_bg, subject).convert("RGB")
        out = io.BytesIO()
        composed.save(out, format="JPEG", quality=72, optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning("Preview generation failed: %s", e)
        return source_jpeg


def _watermarked_white(size: tuple[int, int]) -> Image.Image:
    """Opaque white canvas with diagonal Госфото tiles."""
    base = Image.new("RGBA", size, (255, 255, 255, 255))
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    font = _font(max(22, size[0] // 10))
    measure = ImageDraw.Draw(overlay)
    tw, th = _text_size(measure, WATERMARK_TEXT, font)
    # Dense diagonal grid so white margins always show the mark.
    step_x = max(int(tw * 0.85), 48)
    step_y = max(int(th * 1.35), 40)
    for y in range(-size[1] // 4, size[1] + th, step_y):
        for x in range(-size[0] // 4, size[0] + tw, step_x):
            tile = Image.new("RGBA", (tw + 16, th + 16), (0, 0, 0, 0))
            tile_draw = ImageDraw.Draw(tile)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tile_draw.text(
                    (8 + dx, 8 + dy),
                    WATERMARK_TEXT,
                    fill=WATERMARK_HALO,
                    font=font,
                )
            tile_draw.text((8, 8), WATERMARK_TEXT, fill=WATERMARK_FILL, font=font)
            rotated = tile.rotate(28, expand=True, resample=Image.Resampling.BICUBIC)
            overlay.paste(rotated, (x, y), rotated)
    return Image.alpha_composite(base, overlay)


def _subject_alpha(img: Image.Image, *, tol: int = SUBJECT_WHITE_TOL) -> Image.Image:
    """Soft alpha: opaque on subject, transparent on near-white background."""
    # Distance from white ≈ max channel deficit (same idea as readiness).
    inv = ImageOps.invert(img.convert("RGB"))
    distance = ImageChops.lighter(ImageChops.lighter(inv.getchannel("R"), inv.getchannel("G")), inv.getchannel("B"))
    # > tol → subject (255), else background (0)
    mask = distance.point(lambda v: 255 if v > tol else 0, mode="L")
    # Slight blur so hair/edge doesn't get a hard watermark cut.
    return mask.filter(ImageFilter.GaussianBlur(radius=1.2))


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in (
        "/app/fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    log.error(
        "No TrueType font for watermark; Cyrillic will be broken. "
        "Install fonts-dejavu-core or ship /app/fonts/DejaVuSans.ttf"
    )
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
