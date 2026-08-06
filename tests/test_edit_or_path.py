import numpy as np
import cv2
from unittest.mock import patch

from app import edit as edit_mod


def _png_rgba_red_on_transparent() -> bytes:
    bgra = np.zeros((16, 16, 4), np.uint8)
    bgra[4:12, 4:12, 2] = 255  # R
    bgra[4:12, 4:12, 3] = 255
    ok, buf = cv2.imencode(".png", bgra)
    assert ok
    return buf.tobytes()


def test_or_path_composites_alpha(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "EDIT_BACKEND", "openrouter")
    monkeypatch.setattr(edit_mod.config, "OPENROUTER_API_KEY", "test-key")

    def fake_edit(data, mime="image/jpeg"):
        return _png_rgba_red_on_transparent()

    with patch.object(edit_mod, "edit_selfie", side_effect=fake_edit):
        with patch.object(
            edit_mod, "force_white_background", side_effect=lambda im, tol=52: im
        ):
            out, meta = edit_mod.run_edit_stage(b"jpeg-bytes", "image/jpeg")
    assert meta.get("cutout") == "openrouter"
    assert out.shape[2] == 3
    assert np.all(out[0, 0] == 255)
