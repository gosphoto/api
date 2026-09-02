"""Geometry helpers for passport crop (no MediaPipe)."""

from __future__ import annotations

from . import config


def nudge_crop_off_bottom(top: float, crop_h: float, src_h: float) -> tuple[float, float]:
    """If the crop window runs past the source, shift it up (no white pad below).

    Returns (new_top, shift_px). shift_px is 0 when the window already fits.
    """
    overflow = float(top) + float(crop_h) - float(src_h)
    if overflow <= 0:
        return float(top), 0.0
    return float(top) - overflow, overflow


def crop_width_correction(
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    *,
    override: str | float | None = None,
    corr_max: float | None = None,
) -> float:
    """How much wider than 35×45 the crop window should be before resize.

    >1 → extra side margin → face narrower in the passport frame.
    auto: out_aspect / src_aspect (3:4 Gemini vs 35×45 ≈ 1.066). Already 35×45 → 1.0.
    """
    raw = override if override is not None else config.PASSPORT_CROP_WIDTH_CORR
    raw_s = str(raw).strip().lower()
    hi = float(
        corr_max if corr_max is not None else config.PASSPORT_CROP_WIDTH_CORR_MAX
    )
    if raw_s not in ("auto", ""):
        try:
            return float(min(max(float(raw_s), 1.0), hi))
        except ValueError:
            pass
    src_a = float(src_w) / max(float(src_h), 1.0)
    tgt_a = float(out_w) / max(float(out_h), 1.0)
    corr = tgt_a / max(src_a, 1e-6)
    return float(min(max(corr, 1.0), hi))
