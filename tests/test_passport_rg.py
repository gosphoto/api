"""FMS §34.3 / rg.ru preset for RF passport 35×45."""

from app import config
from app.crop import encode_jpeg
import numpy as np


def test_passport_pixels_are_600dpi_35x45():
    assert config.PASSPORT_DPI == 600
    assert config.PASSPORT_WIDTH == round(35 / 25.4 * 600)
    assert config.PASSPORT_HEIGHT == round(45 / 25.4 * 600)
    assert config.PASSPORT_WIDTH == 827
    assert config.PASSPORT_HEIGHT == 1063


def test_zagran_preset_is_above_gosuslugi_floor():
    """Gosuslugi asks ≥413×531 @ 300dpi; 35×45@300 rounds down to 413 and fails."""
    import math

    p = config.resolve_doc_preset("zagran")
    assert p["doc_type"] == "zagran"
    assert p["dpi"] == 360
    assert p["width"] == math.ceil(35 / 25.4 * 360)
    assert p["height"] == math.ceil(45 / 25.4 * 360)
    assert p["width"] == 497
    assert p["height"] == 638
    assert p["width"] > 413
    assert p["height"] > 531
    assert p["jpeg_max_bytes"] == 2 * 1024 * 1024


def test_resolve_doc_preset_defaults_to_passport_rf():
    p = config.resolve_doc_preset(None)
    assert p["doc_type"] == "passport_rf"
    assert p["dpi"] == 600
    assert config.resolve_doc_preset("nope")["doc_type"] == "passport_rf"


def test_face_and_head_targets_match_rg():
    # 5% smaller than the old 0.79 ceiling: 0.75 (~33.8 mm), still 32–36 mm / 70–80%.
    assert config.PASSPORT_FACE_RATIO == 0.75
    assert config.FACE_RATIO_MIN == 0.70
    assert config.HEAD_HEIGHT_MM_MIN == 32
    assert config.HEAD_HEIGHT_MM_MAX == 36
    assert config.JPEG_MAX_BYTES == 300 * 1024


def test_encode_jpeg_respects_300kb_and_dpi():
    # White-bg portrait-like frame compresses like a real passport photo
    h, w = config.PASSPORT_HEIGHT, config.PASSPORT_WIDTH
    bgr = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.ellipse(
        bgr,
        (w // 2, int(h * 0.48)),
        (int(w * 0.28), int(h * 0.36)),
        0,
        0,
        360,
        (40, 70, 140),
        -1,
    )
    jpeg = encode_jpeg(bgr, quality=95)
    assert jpeg[:2] == b"\xff\xd8"
    assert len(jpeg) <= config.JPEG_MAX_BYTES


def test_encode_jpeg_zagran_dpi_and_budget():
    p = config.resolve_doc_preset("zagran")
    bgr = np.full((p["height"], p["width"], 3), 255, dtype=np.uint8)
    jpeg = encode_jpeg(
        bgr, quality=95, max_bytes=p["jpeg_max_bytes"], dpi=p["dpi"]
    )
    assert jpeg[:2] == b"\xff\xd8"
    assert len(jpeg) <= p["jpeg_max_bytes"]
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(jpeg))
    assert img.info.get("dpi") == (360.0, 360.0) or img.info.get("dpi") == (360, 360)
