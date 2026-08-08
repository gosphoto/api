"""Watermarked preview JPEGs for unpaid result pages."""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("gosphoto-gate")

PREVIEW_MAX_SIDE = 480
WATERMARK_TEXT = "Госфото"


def make_preview_jpeg(source_jpeg: bytes, *, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    """Downscale + diagonal watermark. Falls back to original bytes on error."""
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
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = _font(max(18, img.width // 12))
        tw, th = _text_size(draw, WATERMARK_TEXT, font)
        step_x = max(tw + 40, img.width // 2)
        step_y = max(th + 60, img.height // 3)
        for y in range(-th, img.height + th, step_y):
            for x in range(-tw, img.width + tw, step_x):
                tile = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text((4, 4), WATERMARK_TEXT, fill=(20, 20, 20, 70), font=font)
                rotated = tile.rotate(28, expand=True, resample=Image.Resampling.BICUBIC)
                overlay.paste(rotated, (x, y), rotated)
        composed = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        out = io.BytesIO()
        composed.save(out, format="JPEG", quality=72, optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning("Preview generation failed: %s", e)
        return source_jpeg


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
