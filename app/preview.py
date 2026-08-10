"""Downscaled preview JPEGs for unpaid result pages (no watermark)."""

from __future__ import annotations

import io
import logging

from PIL import Image

log = logging.getLogger("gosphoto-gate")

PREVIEW_MAX_SIDE = 480


def make_preview_jpeg(source_jpeg: bytes, *, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    """Downscale for screen preview. Falls back to original bytes on error."""
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
