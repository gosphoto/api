"""Pre-Gemini Replicate cutout wiring in run_edit_riverflow."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np

from app import edit as edit_mod


def _jpeg(w: int = 32, h: int = 40, color=(30, 60, 90)) -> bytes:
    img = np.full((h, w, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _png_rgba_cutout() -> bytes:
    """Subject in center, transparent border."""
    img = np.zeros((40, 32, 4), np.uint8)
    img[8:32, 8:24, :3] = (20, 40, 60)
    img[8:32, 8:24, 3] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_run_edit_riverflow_applies_pre_cutout(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "PRE_EDIT_REPLICATE_CUTOUT", True)
    monkeypatch.setattr(edit_mod.config, "REPLICATE_API_TOKEN", "r8_test")
    monkeypatch.setattr(edit_mod.config, "FACE_LIGHT_FLATTEN_ENABLED", False)
    monkeypatch.setattr(edit_mod.config, "RIVERFLOW_MODEL", "google/gemini-2.5-flash-image")

    src = np.full((40, 32, 3), 80, np.uint8)
    gemini_out = np.full((40, 32, 3), 250, np.uint8)
    gemini_out[10:30, 10:22] = (50, 50, 50)
    ok, gemini_buf = cv2.imencode(".jpg", gemini_out)
    assert ok

    seen: dict = {}

    def fake_remove(image_bytes: bytes, mime: str = "image/jpeg"):
        seen["cutout_in"] = image_bytes
        return _png_rgba_cutout(), {
            "applied": True,
            "ok": True,
            "model": "851-labs/background-remover",
            "predict_time": 1.2,
            "prediction_id": "pred-test",
        }

    def fake_gemini(image_bytes: bytes, mime: str = "image/jpeg", **kwargs):
        seen["gemini_in"] = image_bytes
        return gemini_buf.tobytes()

    with patch.object(edit_mod, "_decode_image", return_value=src):
        with patch.object(edit_mod, "remove_background", side_effect=fake_remove):
            with patch.object(edit_mod, "edit_selfie_riverflow", side_effect=fake_gemini):
                out, meta = edit_mod.run_edit_riverflow(
                    _jpeg(),
                    mime="image/jpeg",
                    model="google/gemini-2.5-flash-image",
                )

    assert out.shape[2] == 3
    assert meta["pre_cutout"]["applied"] is True
    assert meta["pre_cutout"]["ok"] is True
    assert meta["pre_cutout"]["prediction_id"] == "pred-test"
    assert "cutout_in" in seen and "gemini_in" in seen
    # Pair archive gets the same white-composited JPEG Gemini saw.
    assert isinstance(meta.get("_pre_cutout_jpeg"), (bytes, bytearray))
    assert bytes(meta["_pre_cutout_jpeg"]) == seen["gemini_in"]
    # Gemini must receive the white-composited JPEG, not the raw resize.
    assert seen["gemini_in"] != seen["cutout_in"]
    gemini_decoded = cv2.imdecode(
        np.frombuffer(seen["gemini_in"], np.uint8), cv2.IMREAD_COLOR
    )
    assert gemini_decoded is not None
    # Transparent border became near-white after composite (+ JPEG q=92).
    assert int(gemini_decoded[0, 0].min()) >= 250


def test_run_edit_riverflow_fail_open_on_cutout_error(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "PRE_EDIT_REPLICATE_CUTOUT", True)
    monkeypatch.setattr(edit_mod.config, "REPLICATE_API_TOKEN", "r8_test")
    monkeypatch.setattr(edit_mod.config, "FACE_LIGHT_FLATTEN_ENABLED", False)

    src = np.full((40, 32, 3), 80, np.uint8)
    gemini_out = np.full((40, 32, 3), 255, np.uint8)
    ok, gemini_buf = cv2.imencode(".jpg", gemini_out)
    assert ok

    seen: dict = {}

    def boom(image_bytes: bytes, mime: str = "image/jpeg"):
        raise edit_mod.ReplicateBgError("simulated outage")

    def fake_gemini(image_bytes: bytes, mime: str = "image/jpeg", **kwargs):
        seen["gemini_in"] = image_bytes
        return gemini_buf.tobytes()

    with patch.object(edit_mod, "_decode_image", return_value=src):
        with patch.object(edit_mod, "remove_background", side_effect=boom):
            with patch.object(edit_mod, "edit_selfie_riverflow", side_effect=fake_gemini) as g:
                out, meta = edit_mod.run_edit_riverflow(
                    _jpeg(),
                    mime="image/jpeg",
                    model="google/gemini-2.5-flash-image",
                )

    g.assert_called_once()
    assert meta["pre_cutout"]["ok"] is False
    assert meta["pre_cutout"]["applied"] is False
    assert "simulated outage" in meta["pre_cutout"]["error"]
    assert "_pre_cutout_jpeg" not in meta
    assert out.shape == (40, 32, 3)


def test_run_edit_riverflow_skips_without_token(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "PRE_EDIT_REPLICATE_CUTOUT", True)
    monkeypatch.setattr(edit_mod.config, "REPLICATE_API_TOKEN", "")
    monkeypatch.setattr(edit_mod.config, "FACE_LIGHT_FLATTEN_ENABLED", False)

    src = np.full((40, 32, 3), 80, np.uint8)
    gemini_out = np.full((40, 32, 3), 255, np.uint8)
    ok, gemini_buf = cv2.imencode(".jpg", gemini_out)
    assert ok

    with patch.object(edit_mod, "_decode_image", return_value=src):
        with patch.object(edit_mod, "remove_background") as cut:
            with patch.object(
                edit_mod, "edit_selfie_riverflow", return_value=gemini_buf.tobytes()
            ):
                _out, meta = edit_mod.run_edit_riverflow(
                    _jpeg(),
                    mime="image/jpeg",
                    model="google/gemini-2.5-flash-image",
                )

    cut.assert_not_called()
    assert meta["pre_cutout"]["reason"] == "no_token"
    assert meta["pre_cutout"]["applied"] is False
