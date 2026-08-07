"""Cutout quality lab — try models/postprocess, keep face identity, crop 35×45.

Outputs under tmp-smoke/cutout-lab/ + docs/cutout-lab-2026-08-06.md (caller writes doc).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.crop import encode_jpeg, run_crop_stage
from app.face_restore import restore_face_from_original
from app.whitening import force_white_background

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp-smoke" / "cutout-lab"
MODELS = ROOT / "models"
INPUT = ROOT / "tmp-smoke" / "site-smoke" / "00-input.jpg"

# model_path -> inference size
MODEL_SPECS = {
    "u2netp": (MODELS / "u2netp.onnx", 320),
    "u2net": (MODELS / "u2net.onnx", 320),
    "silueta": (MODELS / "silueta.onnx", 320),
    "isnet": (MODELS / "isnet-general-use.onnx", 1024),
}


@dataclass
class Score:
    name: str
    seconds: float
    fringe_p90: float
    fringe_mean: float
    shoulder_canny: float
    face_l1: float
    white_ok: bool
    pass_geo: bool
    notes: str = ""


_SESSIONS: dict[str, ort.InferenceSession] = {}


def _session(path: Path) -> ort.InferenceSession:
    key = str(path)
    if key not in _SESSIONS:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _SESSIONS[key] = ort.InferenceSession(
            key, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[key]


def onnx_mask(bgr: np.ndarray, path: Path, size: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    im = (im - mean) / std
    tensor = im.transpose(2, 0, 1)[None, ...].astype(np.float32)
    sess = _session(path)
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    pred = out[0][0] if out.ndim == 4 else out[0]
    pred = pred.astype(np.float32)
    pred = (pred - pred.min()) / (float(pred.max() - pred.min()) + 1e-8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)


def largest_cc(m8: np.ndarray) -> np.ndarray:
    hard = (m8 > 120).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if num <= 1:
        return m8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((labels == largest).astype(np.uint8), np.ones((15, 15), np.uint8))
    return cv2.bitwise_and(m8, keep * 255)


def post_soft(mask: np.ndarray, erode: int = 1, sigma: float = 1.2) -> np.ndarray:
    m = np.clip((mask - 0.12) / 0.58, 0, 1)
    m8 = (m * 255).astype(np.uint8)
    m8 = cv2.bilateralFilter(m8, 7, 40, 7)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m8 = largest_cc(m8)
    if erode > 0:
        m8 = cv2.erode(m8, np.ones((3, 3), np.uint8), iterations=erode)
    m8 = cv2.GaussianBlur(m8, (0, 0), sigmaX=sigma)
    a = np.clip(m8.astype(np.float32) / 255.0, 0, 1)
    return a * a * (3 - 2 * a)


def post_distance(mask: np.ndarray, band: float = 4.0) -> np.ndarray:
    """Hard threshold + distance-transform soft edge (smooth silhouette)."""
    m = np.clip((mask - 0.35) / 0.40, 0, 1)
    hard = (m > 0.5).astype(np.uint8) * 255
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    hard = largest_cc(hard)
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    din = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(255 - hard, cv2.DIST_L2, 5)
    a = din / (din + dout + 1e-6)
    # compress soft band
    a = np.clip((a - 0.5) * (band * 0.5) + 0.5, 0, 1)
    return a.astype(np.float32)


def post_guided(mask: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Edge-aware refine using joint bilateral on mask guided by luma."""
    m = post_soft(mask, erode=1, sigma=0.8)
    m8 = (m * 255).astype(np.uint8)
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # OpenCV ximgproc may be missing; emulate with bilateral on stacked
    refined = cv2.bilateralFilter(m8, 9, 50, 50)
    # Pull mask toward guide edges
    edges = cv2.Canny(guide, 60, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(refined, (0, 0), 1.0)
    out = refined.copy()
    out[edges > 0] = soft[edges > 0]
    a = out.astype(np.float32) / 255.0
    return a * a * (3 - 2 * a)


def composite(bgr: np.ndarray, a: np.ndarray, decontam: float = 0.55) -> np.ndarray:
    a = np.clip(a, 0, 1).astype(np.float32)
    a3 = a[:, :, None]
    fg = bgr.astype(np.float32)
    white = np.full_like(fg, 255.0)
    soft = ((a > 0.04) & (a < 0.92)).astype(np.float32)[:, :, None]
    fg = fg * (1.0 - decontam * soft) + white * (decontam * soft)
    return np.clip(fg * a3 + white * (1.0 - a3), 0, 255).astype(np.uint8)


def face_l1(orig: np.ndarray, out: np.ndarray) -> float:
    """Mean abs diff on face oval — lower is better (identity)."""
    from app.face_restore import _face_mask, _landmarks_xy

    h, w = out.shape[:2]
    o = cv2.resize(orig, (w, h))
    lm = _landmarks_xy(o)
    if lm is None:
        return 999.0
    m_face, m_hair = _face_mask((h, w), lm)
    m = np.maximum(m_face, m_hair)
    diff = np.abs(o.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
    vals = diff[m > 0.5]
    return float(vals.mean()) if vals.size else 999.0


def fringe_stats(im: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    person = gray < 245
    edge = (
        cv2.morphologyEx(
            person.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        > 0
    )
    f = im.astype(np.float32)
    chroma = np.linalg.norm(f - f.mean(axis=2)[:, :, None], axis=2)
    fringe = chroma[edge]
    roi = im[int(im.shape[0] * 0.7) :, : int(im.shape[1] * 0.35)]
    canny = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 80, 160)
    if fringe.size == 0:
        return 0.0, 0.0, float(canny.sum() / 255)
    return float(fringe.mean()), float(np.percentile(fringe, 90)), float(canny.sum() / 255)


def panel(im: np.ndarray, title: str, tw: int = 220) -> np.ndarray:
    h = int(im.shape[0] * tw / im.shape[1])
    x = cv2.resize(im, (tw, h))
    bar = np.full((34, tw, 3), 28, np.uint8)
    cv2.putText(bar, title[:28], (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([bar, x])


def run_variant(
    name: str,
    bgr: np.ndarray,
    mask: np.ndarray,
    post: str,
    face_restore: bool,
    decontam: float,
) -> tuple[np.ndarray, np.ndarray, Score]:
    t0 = time.time()
    if post == "soft1":
        a = post_soft(mask, erode=1, sigma=1.2)
    elif post == "soft2":
        a = post_soft(mask, erode=2, sigma=1.6)
    elif post == "dist":
        a = post_distance(mask, band=5.0)
    elif post == "guided":
        a = post_guided(mask, bgr)
    else:
        raise ValueError(post)

    edited = composite(bgr, a, decontam=decontam)
    note = ""
    if face_restore:
        edited2, ok = restore_face_from_original(bgr, edited)
        edited = edited2
        note = f"face_restore={ok}"
    edited = force_white_background(edited, tol=48)
    cropped, _, comp = run_crop_stage(edited)
    dt = time.time() - t0
    fm, fp90, shoulder = fringe_stats(cropped)
    fl1 = face_l1(bgr, cropped)
    score = Score(
        name=name,
        seconds=round(dt, 2),
        fringe_p90=round(fp90, 2),
        fringe_mean=round(fm, 2),
        shoulder_canny=round(shoulder, 1),
        face_l1=round(fl1, 2),
        white_ok=bool((comp.get("checks") or {}).get("bg_white_ok")),
        pass_geo=bool(comp.get("pass")),
        notes=note,
    )
    return edited, cropped, score


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(INPUT))
    assert bgr is not None, INPUT
    # work at process size
    h, w = bgr.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    scores: list[Score] = []
    thumbs: list[np.ndarray] = []

    # cache masks per model
    masks: dict[str, np.ndarray] = {}
    for mid, (path, size) in MODEL_SPECS.items():
        if not path.is_file():
            print("skip missing", path)
            continue
        print("mask", mid, "...")
        t0 = time.time()
        masks[mid] = onnx_mask(bgr, path, size)
        print(f"  {mid} mask {time.time()-t0:.2f}s")

    variants: list[tuple[str, str, str, bool, float]] = []
    for mid in masks:
        for post in ("soft1", "soft2", "dist", "guided"):
            for fr in (True, False):
                # face restore ON is required path for identity; still compare OFF
                decontam = 0.65 if post in ("soft2", "dist") else 0.5
                tag = f"{mid}_{post}_{'fr' if fr else 'nfr'}"
                variants.append((tag, mid, post, fr, decontam))

    # Priority order: evaluate promising first but run all
    for tag, mid, post, fr, decontam in variants:
        print("run", tag)
        edited, cropped, score = run_variant(
            tag, bgr, masks[mid], post, fr, decontam
        )
        scores.append(score)
        cv2.imwrite(str(OUT / f"{tag}_full.jpg"), edited)
        Path(OUT / f"{tag}_crop.jpg").write_bytes(encode_jpeg(cropped))
        thumbs.append(panel(cropped, tag))

    # ranking: prefer low fringe_p90 + low face_l1 + pass + white
    def rank_key(s: Score) -> tuple:
        return (
            0 if s.pass_geo and s.white_ok else 1,
            s.face_l1 + s.fringe_p90 * 0.35,
            s.fringe_p90,
            s.seconds,
        )

    ranked = sorted(scores, key=rank_key)
    best = ranked[0]
    print("BEST", best)

    # board of top 8 crops
    top = ranked[:8]
    top_panels = []
    for s in top:
        im = cv2.imread(str(OUT / f"{s.name}_crop.jpg"))
        if im is None:
            continue
        top_panels.append(panel(im, f"{s.name}\nf{s.face_l1} p{s.fringe_p90}", 200))
    if top_panels:
        # 2 rows
        row1 = np.hstack(top_panels[:4]) if len(top_panels) >= 4 else np.hstack(top_panels)
        if len(top_panels) > 4:
            row2 = np.hstack(top_panels[4:8])
            # pad widths
            if row2.shape[1] < row1.shape[1]:
                pad = np.full((row2.shape[0], row1.shape[1] - row2.shape[1], 3), 255, np.uint8)
                row2 = np.hstack([row2, pad])
            board = np.vstack([row1, row2])
        else:
            board = row1
        cv2.imwrite(str(OUT / "board-top8.jpg"), board)

    # input | best full | best crop
    best_full = cv2.imread(str(OUT / f"{best.name}_full.jpg"))
    best_crop = cv2.imread(str(OUT / f"{best.name}_crop.jpg"))
    compare = np.hstack(
        [
            panel(bgr, "INPUT", 280),
            panel(best_full, f"BEST full {best.name}", 280),
            panel(best_crop, "BEST crop 35x45", 280),
        ]
    )
    cv2.imwrite(str(OUT / "board-best.jpg"), compare)

    Path(OUT / "scores.json").write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "best": asdict(best),
                "ranked": [asdict(s) for s in ranked],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
