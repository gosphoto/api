"""Passport crop: roll-align → fit face/margins → 35×45 @ 600dpi (FMS §34.3).

Expects a white-background portrait (local cutout). No generative rewrite.
Post-crop Gemini cleanup is off by default: it widens face oval 3–8%.
PASSPORT_CROP_WIDTH_CORR default 1 (off); auto squeezed 3:4 too narrow.
Retries crown/face-ratio variants until compliance is closest to pass.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .baldness import analyze_baldness
from .compliance import measure_compliance
from .compose_bg import composite_on_white
from .crop_geom import crop_width_correction
from .gate import _landmarker
from .openrouter import POST_CROP_CLEANUP_PROMPT, edit_selfie_riverflow
from .whitening import force_white_background

log = logging.getLogger("gosphoto-gate")

_LEFT_EYE = 33
_RIGHT_EYE = 263
_CHIN = 152
_FOREHEAD = 10


def _crop_once(
    bgr: np.ndarray,
    *,
    crown_factor: float,
    face_ratio: float,
    top_margin: float,
    out_w: int | None = None,
    out_h: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        raise ValueError("no_face")

    lm = result.face_landmarks[0]
    h, w = bgr.shape[:2]

    le = lm[_LEFT_EYE]
    re = lm[_RIGHT_EYE]
    chin = lm[_CHIN]
    forehead = lm[_FOREHEAD]

    dx = (re.x - le.x) * w
    dy = (re.y - le.y) * h
    roll_deg = math.degrees(math.atan2(dy, dx))

    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, roll_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    rgb2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    result2 = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2))
    lm2 = result2.face_landmarks[0] if result2.face_landmarks else lm
    rotated_src = rotated

    le = lm2[_LEFT_EYE]
    re = lm2[_RIGHT_EYE]
    chin = lm2[_CHIN]
    forehead = lm2[_FOREHEAD]

    chin_y = chin.y * h
    forehead_y = forehead.y * h
    crown_y = forehead_y - crown_factor * (chin_y - forehead_y)
    face_h = max(chin_y - crown_y, 1.0)
    mid_x = ((le.x + re.x) / 2) * w

    out_w = int(out_w if out_w is not None else config.PASSPORT_WIDTH)
    out_h = int(out_h if out_h is not None else config.PASSPORT_HEIGHT)
    target_face = face_ratio * out_h
    scale = target_face / face_h

    crop_h = out_h / scale
    width_corr = crop_width_correction(w, h, out_w, out_h)
    crop_w = out_w / scale * width_corr
    top = crown_y - top_margin * crop_h
    left = mid_x - crop_w / 2

    pad = int(max(crop_w, crop_h)) + 8
    canvas = cv2.copyMakeBorder(
        rotated_src,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    left_p = left + pad
    top_p = top + pad
    x0 = max(0, int(round(left_p)))
    y0 = max(0, int(round(top_p)))
    x1 = min(canvas.shape[1], int(round(left_p + crop_w)))
    y1 = min(canvas.shape[0], int(round(top_p + crop_h)))
    patch = canvas[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("empty_crop")

    ph, pw = patch.shape[:2]
    interp = cv2.INTER_AREA if (pw > out_w or ph > out_h) else cv2.INTER_CUBIC
    out = cv2.resize(patch, (out_w, out_h), interpolation=interp)
    metrics = {
        "roll_corrected_deg": round(roll_deg, 2),
        "width": out_w,
        "height": out_h,
        "face_ratio_target": face_ratio,
        "crown_factor": crown_factor,
        "top_margin_target": top_margin,
        "face_ratio_est": round(float(face_h * scale / out_h), 3),
        "width_corr": round(width_corr, 4),
        "src_aspect": round(w / max(h, 1), 4),
    }
    return out, metrics


# Crown gap → head target. Short hair / buzz: same 0.75 head looks like a
# huge face oval (returned 2026-09-02: head 33.7 mm, face_only 0.58).
SHORT_HAIR_GAP = 0.30
FULL_HAIR_GAP = 0.38
SHORT_HAIR_FACE_RATIO = 0.72
FACE_ONLY_SOFT_MAX = 0.56


def face_target_from_gap(gap_ratio: float | None) -> float:
    """Head ratio aim: ≤0.30 gap → 0.72, ≥0.38 → PASSPORT_FACE_RATIO, else lerp."""
    hi = float(config.PASSPORT_FACE_RATIO)
    lo = float(SHORT_HAIR_FACE_RATIO)
    if gap_ratio is None:
        return hi
    gap = float(gap_ratio)
    if gap <= SHORT_HAIR_GAP:
        return lo
    if gap >= FULL_HAIR_GAP:
        return hi
    t = (gap - SHORT_HAIR_GAP) / (FULL_HAIR_GAP - SHORT_HAIR_GAP)
    return round(lo + t * (hi - lo), 3)


def _score_compliance(
    comp: dict[str, Any],
    *,
    face_target: float | None = None,
) -> float:
    """Higher is better. Passing crops are not equal: prefer aim + compact face."""
    aim = float(face_target if face_target is not None else config.PASSPORT_FACE_RATIO)
    score = 1000.0 if comp.get("pass") else 0.0
    if not comp.get("pass"):
        checks = comp.get("checks") or {}
        for k in (
            "size_ok",
            "face_oval_ok",  # informational 70–80%; not a hard gate
            "head_height_mm_ok",
            "head_width_mm_ok",
            "face_ratio_ok",
            "top_margin_ok",
            "bg_white_ok",
            "single_face_ok",
        ):
            if checks.get(k):
                score += 100.0
    fr = float(comp.get("face_ratio") or 0)
    tm = float(comp.get("top_margin") or 0)
    fo = float(comp.get("face_only") or 0)
    score -= abs(fr - aim) * 200
    score -= abs(tm - config.PASSPORT_TOP_MARGIN) * 300
    if fo > FACE_ONLY_SOFT_MAX:
        score -= (fo - FACE_ONLY_SOFT_MAX) * 800
    return score


def crop_passport(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible single-shot crop (default geometry)."""
    return _crop_once(
        bgr,
        crown_factor=0.45,
        face_ratio=config.PASSPORT_FACE_RATIO,
        top_margin=config.PASSPORT_TOP_MARGIN,
    )


def _crop_attempts(bald: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Geometry grid; head target from crown gap, not the is_bald flag."""
    face_r = face_target_from_gap(bald.get("gap_ratio"))
    top_m = config.PASSPORT_TOP_MARGIN
    default: list[tuple[float, float, float]] = [
        (0.45, face_r, top_m),
        (0.50, face_r, 0.10),
        (0.55, max(0.70, face_r - 0.02), 0.09),
        (0.40, face_r, 0.11),
        (0.60, max(0.70, face_r - 0.02), 0.10),
        (0.35, min(0.80, face_r + 0.02), 0.10),
    ]
    hinted = float(bald.get("crown_factor") or (0.22 if bald.get("is_bald") else 0.45))
    hinted = float(np.clip(hinted, 0.14, 0.65))

    priority: list[tuple[float, float, float]] = [
        (hinted, face_r, top_m),
        (hinted, face_r, 0.10),
        (float(np.clip(hinted - 0.04, 0.14, 0.65)), face_r, 0.10),
        (float(np.clip(hinted + 0.04, 0.14, 0.65)), max(0.70, face_r - 0.02), 0.11),
        (hinted, max(0.70, face_r - 0.02), 0.11),
    ]
    gap = bald.get("gap_ratio")
    short = bald.get("is_bald") or (
        gap is not None and float(gap) <= SHORT_HAIR_GAP
    )
    if short:
        priority.extend(
            [
                (0.22, face_r, 0.10),
                (0.18, face_r, 0.09),
                (0.26, max(0.70, face_r - 0.02), 0.11),
                (0.30, face_r, 0.10),
                (0.35, max(0.70, face_r - 0.02), 0.10),
            ]
        )

    seen: set[tuple[float, float, float]] = set()
    out: list[tuple[float, float, float]] = []
    for item in priority + default:
        key = (round(item[0], 3), round(item[1], 3), round(item[2], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _finalize_crop(cropped: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Optional second Gemini pass (off by default — widens faces), then local whitening."""
    model = config.POST_CROP_CLEANUP_MODEL or config.RIVERFLOW_MODEL
    meta: dict[str, Any] = {"applied": False, "model": model}
    cleaned = cropped

    if config.POST_CROP_CLEANUP_ENABLED and config.OPENROUTER_API_KEY:
        try:
            ok, buf = cv2.imencode(
                ".jpg",
                cropped,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            if not ok:
                raise RuntimeError("post_crop_encode_failed")
            raw = edit_selfie_riverflow(
                buf.tobytes(),
                "image/jpeg",
                model=model,
                prompt=POST_CROP_CLEANUP_PROMPT,
            )
            decoded = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if decoded is None:
                raise RuntimeError("post_crop_decode_failed")
            if decoded.ndim == 2:
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
            else:
                decoded = composite_on_white(decoded)
            cleaned = cv2.resize(
                decoded,
                (cropped.shape[1], cropped.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            meta["applied"] = True
        except Exception as exc:
            meta["error"] = str(exc)[:200]
            log.warning("Post-crop background cleanup failed; using crop: %s", exc)
    else:
        meta["reason"] = (
            "disabled"
            if not config.POST_CROP_CLEANUP_ENABLED
            else "openrouter_key_missing"
        )

    return force_white_background(cleaned, tol=55, soften=True), meta


def _finalize_with_compliance(
    cropped: np.ndarray,
    baseline_comp: dict[str, Any],
    *,
    out_w: int,
    out_h: int,
    dpi_target: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Reject a model cleanup if it turns a passing crop into a failing one."""
    finalized, cleanup_meta = _finalize_crop(cropped)
    comp = measure_compliance(
        finalized,
        expected_width=out_w,
        expected_height=out_h,
        dpi_target=dpi_target,
    )
    if (
        cleanup_meta.get("applied")
        and baseline_comp.get("pass")
        and not comp.get("pass")
    ):
        failed_checks = [
            key for key, passed in (comp.get("checks") or {}).items() if not passed
        ]
        finalized = force_white_background(cropped, tol=55, soften=True)
        comp = measure_compliance(
            finalized,
            expected_width=out_w,
            expected_height=out_h,
            dpi_target=dpi_target,
        )
        cleanup_meta.update(
            {
                "applied": False,
                "rejected": True,
                "reason": "compliance_regression",
                "failed_checks": failed_checks,
            }
        )
        log.warning(
            "Rejected post-crop cleanup because compliance regressed: %s",
            failed_checks,
        )
    return finalized, cleanup_meta, comp


def run_crop_stage(
    bgr: np.ndarray,
    *,
    width: int | None = None,
    height: int | None = None,
    dpi: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """
    White-bg portrait → 35×45 passport BGR + metrics + compliance.

    Tries several crown/face geometries and keeps the best compliance result.
    Targets Gosuslugi 70–80% / FMS head 32–36 mm (crop aim 0.75).
    Output pixel size follows width/height (default RF 600 dpi).
    """
    out_w = int(width if width is not None else config.PASSPORT_WIDTH)
    out_h = int(height if height is not None else config.PASSPORT_HEIGHT)
    dpi_target = int(dpi if dpi is not None else config.PASSPORT_DPI)
    bald = analyze_baldness(bgr).as_dict()
    face_aim = face_target_from_gap(bald.get("gap_ratio"))
    attempts = _crop_attempts(bald)

    best: tuple[np.ndarray, dict[str, Any], dict[str, Any], float] | None = None
    errors: list[str] = []

    for crown_f, face_r, top_m in attempts:
        try:
            cropped, metrics = _crop_once(
                bgr,
                crown_factor=crown_f,
                face_ratio=face_r,
                top_margin=top_m,
                out_w=out_w,
                out_h=out_h,
            )
            cropped = force_white_background(cropped, tol=55, soften=False)
            comp = measure_compliance(
                cropped,
                expected_width=out_w,
                expected_height=out_h,
                dpi_target=dpi_target,
            )
            score = _score_compliance(comp, face_target=face_aim)
            metrics = {
                **metrics,
                "attempt_score": round(score, 2),
                "baldness": bald,
                "dpi": dpi_target,
            }
            if best is None or score > best[3]:
                best = (cropped, metrics, comp, score)
            # Soft compliance pass can undershoot top field (~3 mm). Keep
            # searching until top_margin is near MVD 5±1 mm (≈0.09–0.13).
            tm = float(comp.get("top_margin") or 0.0)
            fo = float(comp.get("face_only") or 0.0)
            if comp.get("pass") and tm >= 0.09 and fo <= FACE_ONLY_SOFT_MAX:
                break
        except Exception as e:
            errors.append(f"{crown_f}/{face_r}: {e}")

    if best is None:
        # Last-resort center crop to 35×45
        h, w = bgr.shape[:2]
        target_ratio = out_w / out_h
        cur = w / max(h, 1)
        if cur > target_ratio:
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            patch = bgr[:, x0 : x0 + new_w]
        else:
            new_h = int(w / target_ratio)
            y0 = (h - new_h) // 2
            patch = bgr[y0 : y0 + new_h, :]
        cropped = cv2.resize(
            patch,
            (out_w, out_h),
            interpolation=cv2.INTER_AREA,
        )
        cropped = force_white_background(cropped, tol=55, soften=False)
        baseline_comp = measure_compliance(
            cropped,
            expected_width=out_w,
            expected_height=out_h,
            dpi_target=dpi_target,
        )
        cropped, cleanup_meta, comp = _finalize_with_compliance(
            cropped,
            baseline_comp,
            out_w=out_w,
            out_h=out_h,
            dpi_target=dpi_target,
        )
        metrics = {
            "fallback": True,
            "errors": errors[:4],
            "baldness": bald,
            "dpi": dpi_target,
            "width": out_w,
            "height": out_h,
            "post_crop_cleanup": cleanup_meta,
        }
        return cropped, metrics, comp

    cropped, metrics, baseline_comp, _ = best
    cropped, cleanup_meta, comp = _finalize_with_compliance(
        cropped,
        baseline_comp,
        out_w=out_w,
        out_h=out_h,
        dpi_target=dpi_target,
    )
    if errors:
        metrics["skipped_errors"] = errors[:3]
    metrics["baldness"] = bald
    metrics["dpi"] = dpi_target
    metrics["post_crop_cleanup"] = cleanup_meta
    return cropped, metrics, comp


def encode_jpeg(
    bgr: np.ndarray,
    quality: int | None = None,
    max_bytes: int | None = None,
    dpi: int | None = None,
) -> bytes:
    """JPEG with DPI metadata; shrink quality until under max_bytes."""
    from io import BytesIO

    from PIL import Image

    q0 = int(quality if quality is not None else config.JPEG_QUALITY)
    limit = int(max_bytes if max_bytes is not None else config.JPEG_MAX_BYTES)
    dpi_val = int(dpi if dpi is not None else config.PASSPORT_DPI)
    dpi_tuple = (dpi_val, dpi_val)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    best = b""
    for q in list(range(q0, 54, -4)) + [50]:
        buf = BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=max(1, q),
            dpi=dpi_tuple,
            optimize=True,
        )
        best = buf.getvalue()
        if len(best) <= limit:
            return best
    return best
