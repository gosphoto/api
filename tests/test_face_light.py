"""CPU face-light flatten: even out one-sided cheek L without a model call."""

from __future__ import annotations

import numpy as np

from app.face_light import flatten_face_lighting
from app.face_protect import _FACE_OVAL_IDX


def _oval_landmarks(cx: float = 80.0, cy: float = 90.0, rx: float = 40.0, ry: float = 55.0):
    pts = np.zeros((478, 2), np.float32)
    pts[:] = [cx, cy]
    for i, idx in enumerate(_FACE_OVAL_IDX):
        ang = 2 * np.pi * i / len(_FACE_OVAL_IDX)
        pts[idx] = [cx + rx * np.cos(ang), cy + ry * np.sin(ang)]
    pts[1] = [cx, cy]  # nose tip
    return pts


def _uneven_portrait(w: int = 160, h: int = 200) -> np.ndarray:
    """White bg + skin oval with strong left-bright / right-dark gradient."""
    img = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cx, cy, rx, ry = 80.0, 90.0, 40.0, 55.0
    oval = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    # Base skin; left half brighter than right.
    skin = np.zeros((h, w, 3), np.uint8)
    skin[:] = (140, 160, 200)  # BGR-ish warm
    t = np.clip((xx - (cx - rx)) / (2 * rx), 0, 1)
    # Right side (~t=1) ~60 darker in all channels.
    darken = (t * 70).astype(np.uint8)
    for c in range(3):
        ch = skin[:, :, c].astype(np.int16) - darken
        skin[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    img[oval] = skin[oval]
    return img


def test_flatten_reduces_left_right_delta(monkeypatch):
    import app.face_light as fl

    pts = _oval_landmarks()
    monkeypatch.setattr(fl, "_landmarks_xy", lambda bgr: pts.copy())
    # face_protect_mask uses its own landmarker — patch there too
    import app.face_protect as fp

    monkeypatch.setattr(fp, "_landmarks_xy", lambda bgr: pts.copy())

    src = _uneven_portrait()
    out, meta = flatten_face_lighting(src, delta_min=12.0, max_gain=50.0, max_iters=3)
    assert meta["applied"] is True
    assert "before" in meta and "after" in meta
    assert abs(meta["before"]["delta"]) >= 12
    assert abs(meta["after"]["delta"]) < abs(meta["before"]["delta"])
    assert abs(meta["after"]["delta"]) < 12


def test_flatten_noop_when_already_even(monkeypatch):
    import app.face_light as fl
    import app.face_protect as fp

    pts = _oval_landmarks()
    monkeypatch.setattr(fl, "_landmarks_xy", lambda bgr: pts.copy())
    monkeypatch.setattr(fp, "_landmarks_xy", lambda bgr: pts.copy())

    img = np.full((200, 160, 3), 255, np.uint8)
    yy, xx = np.ogrid[:200, :160]
    oval = ((xx - 80) / 40) ** 2 + ((yy - 90) / 55) ** 2 <= 1.0
    img[oval] = (150, 170, 200)
    out, meta = flatten_face_lighting(img, delta_min=12.0)
    assert meta["applied"] is False
    assert meta.get("reason") == "already_even"
    assert np.array_equal(out, img)


def test_flatten_noop_without_face(monkeypatch):
    import app.face_light as fl

    monkeypatch.setattr(fl, "_landmarks_xy", lambda bgr: None)
    img = np.zeros((40, 40, 3), np.uint8)
    out, meta = flatten_face_lighting(img)
    assert meta["applied"] is False
    assert meta["reason"] == "no_face"
    assert np.array_equal(out, img)
