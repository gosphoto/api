"""HEIC/HEIF decode — iPhone often uploads HEIC with a .jpg filename."""

from pathlib import Path

from app.gate import _decode_image

_TINY_HEIC = Path(__file__).resolve().parent / "fixtures" / "heic" / "tiny.heic"


def test_decode_heic_fixture():
    data = _TINY_HEIC.read_bytes()
    assert data[4:8] == b"ftyp"
    bgr = _decode_image(data)
    assert bgr is not None
    assert bgr.ndim == 3
    assert bgr.shape[2] == 3
    assert bgr.shape[0] > 0 and bgr.shape[1] > 0


def test_decode_heic_misnamed_as_jpg():
    """Prod case: HEIC bytes, client filename ends with .jpg."""
    data = _TINY_HEIC.read_bytes()
    # Filename is irrelevant to _decode_image — content only.
    bgr = _decode_image(data)
    assert bgr is not None
