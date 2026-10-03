"""Preview JPEGs for unpaid result pages.

Small result-card preview: downscale only (no watermark).
Full-size view page: tiled watermark so screenshots are not upload-ready.
"""

from __future__ import annotations

import io
import logging
import math

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("gosphoto-gate")

PREVIEW_MAX_SIDE = 480
VIEW_JPEG_QUALITY = 85
WATERMARK_TEXT = "ГОСФОТО"


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def apply_watermark(img: Image.Image) -> Image.Image:
    """Tile a semi-transparent diagonal mark so screenshots are not upload-ready."""
    base = img.convert("RGBA")
    w, h = base.size

    font_size = max(18, min(w, h) // 11)
    font = _load_font(font_size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), WATERMARK_TEXT, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x, pad_y = max(28, w // 8), max(36, h // 7)
    tile = Image.new("RGBA", (tw + pad_x, th + pad_y), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    x0, y0 = pad_x // 2, pad_y // 2
    for dx, dy, fill in (
        (1, 1, (25, 25, 25, 70)),
        (0, 0, (255, 255, 255, 110)),
    ):
        tile_draw.text((x0 + dx, y0 + dy), WATERMARK_TEXT, font=font, fill=fill)

    rotated = tile.rotate(-32, expand=True, resample=Image.Resampling.BICUBIC)
    rw, rh = rotated.size
    step_x = max(1, int(rw * 0.72))
    step_y = max(1, int(rh * 0.72))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    diag = int(math.hypot(w, h)) + max(rw, rh)
    y = -diag
    row = 0
    while y < h + diag:
        x = -diag + (row % 2) * (step_x // 2)
        while x < w + diag:
            overlay.alpha_composite(rotated, (x, y))
            x += step_x
        y += step_y
        row += 1

    draw = ImageDraw.Draw(overlay)
    strip_h = max(22, h // 14)
    draw.rectangle((0, h - strip_h, w, h), fill=(20, 24, 32, 155))
    strip_font = _load_font(max(12, strip_h - 8))
    label = "Превью · скачайте после оплаты"
    lb = draw.textbbox((0, 0), label, font=strip_font)
    lw, lh = lb[2] - lb[0], lb[3] - lb[1]
    draw.text(
        ((w - lw) / 2, h - strip_h + max(2, (strip_h - lh) // 2)),
        label,
        font=strip_font,
        fill=(255, 255, 255, 230),
    )

    return Image.alpha_composite(base, overlay).convert("RGB")


def make_preview_jpeg(source_jpeg: bytes, *, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    """Downscale for screen preview. Falls back to original on error."""
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
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=72, optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning("Preview generation failed: %s", e)
        return source_jpeg


def make_view_jpeg(source_jpeg: bytes) -> bytes:
    """Full-size JPEG with watermark for the unpaid view page."""
    if not source_jpeg:
        return b""
    try:
        img = Image.open(io.BytesIO(source_jpeg)).convert("RGB")
        img = apply_watermark(img)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=VIEW_JPEG_QUALITY, optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning("View watermark generation failed: %s", e)
        return source_jpeg
