import numpy as np
import cv2
from unittest.mock import patch

from app import edit as edit_mod


def test_openrouter_backend_uses_local_person_no_shift(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "EDIT_BACKEND", "openrouter")
    monkeypatch.setattr(edit_mod.config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EDIT_USE_OR_PIXELS", "0")

    src = np.full((32, 32, 3), 40, np.uint8)
    local = np.full((32, 32, 3), 255, np.uint8)
    local[8:24, 8:24] = (40, 40, 40)

    with patch.object(edit_mod, "_decode_image", return_value=src):
        with patch.object(edit_mod, "white_background_local", return_value=local):
            with patch.object(
                edit_mod,
                "force_white_background",
                side_effect=lambda im, tol=52: im,
            ):
                with patch.object(edit_mod, "edit_selfie") as or_mock:
                    out, meta = edit_mod.run_edit_stage(b"jpeg-bytes", "image/jpeg")
    or_mock.assert_not_called()
    assert meta.get("face_protect", {}).get("composite") == "local_person_white_bg"
    assert out.shape == (32, 32, 3)
    assert np.all(out[0, 0] == 255)


def test_local_preferred_over_openrouter(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "EDIT_BACKEND", "local")
    monkeypatch.setattr(edit_mod.config, "OPENROUTER_API_KEY", "test-key")

    local_img = np.full((20, 20, 3), 200, np.uint8)
    with patch.object(edit_mod, "_decode_image", return_value=local_img):
        with patch.object(
            edit_mod,
            "edit_selfie_local",
            return_value=(local_img, {"cutout": "mediapipe", "width": 20, "height": 20}),
        ) as local_mock:
            with patch.object(edit_mod, "edit_selfie") as or_mock:
                out, meta = edit_mod.run_edit_stage(b"jpeg-bytes", "image/jpeg")
    assert meta.get("cutout") == "mediapipe"
    assert meta.get("model") == "mediapipe"
    local_mock.assert_called_once()
    or_mock.assert_not_called()
    assert out.shape == (20, 20, 3)
