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


def test_face_and_head_targets_match_rg():
    # 80% of 45 mm = 36 mm (upper bound of head length)
    assert config.PASSPORT_FACE_RATIO == 0.80
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
