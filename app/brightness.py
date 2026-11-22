# app/brightness.py
from __future__ import annotations

from PIL import Image, ImageEnhance

DEFAULT_BRIGHTNESS = 1.0
MIN_BRIGHTNESS = 0.7
MAX_BRIGHTNESS = 1.35
BRIGHTNESS_EPS = 0.01


def parse_brightness(value: object) -> float:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError("brightness must be a number") from e
    if not (MIN_BRIGHTNESS <= n <= MAX_BRIGHTNESS):
        raise ValueError(
            f"brightness must be between {MIN_BRIGHTNESS} and {MAX_BRIGHTNESS}"
        )
    return round(n, 4)


def is_dirty(brightness: float) -> bool:
    return abs(float(brightness) - DEFAULT_BRIGHTNESS) >= BRIGHTNESS_EPS


def apply_brightness(photo: Image.Image, brightness: float) -> Image.Image:
    factor = float(brightness)
    if abs(factor - DEFAULT_BRIGHTNESS) < 1e-6:
        return photo
    return ImageEnhance.Brightness(photo.convert("RGB")).enhance(factor)


DEFAULT_CONTRAST = 1.0
MIN_CONTRAST = 0.7
MAX_CONTRAST = 1.35
CONTRAST_EPS = 0.01


def parse_contrast(value: object) -> float:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError("contrast must be a number") from e
    if not (MIN_CONTRAST <= n <= MAX_CONTRAST):
        raise ValueError(
            f"contrast must be between {MIN_CONTRAST} and {MAX_CONTRAST}"
        )
    return round(n, 4)


def is_contrast_dirty(contrast: float) -> bool:
    return abs(float(contrast) - DEFAULT_CONTRAST) >= CONTRAST_EPS


def apply_contrast(photo: Image.Image, contrast: float) -> Image.Image:
    factor = float(contrast)
    if abs(factor - DEFAULT_CONTRAST) < 1e-6:
        return photo
    return ImageEnhance.Contrast(photo.convert("RGB")).enhance(factor)


def apply_adjust(
    photo: Image.Image, brightness: float, contrast: float
) -> Image.Image:
    return apply_contrast(apply_brightness(photo, brightness), contrast)