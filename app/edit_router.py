"""Pick Riverflow Pro vs Gemini from cheap CPU cues on the input selfie."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from . import config


def _blur_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


GEMINI_FLASH_IMAGE = "google/gemini-2.5-flash-image"
RIVERFLOW_PRO = "sourceful/riverflow-v2.5-pro"

# Top-corner wall: her selfie ~231; dark rooms sit well below this.
LIGHT_BG_LUMA_MIN = 185.0
# Gate floor is 10; Pro is wasted on mushy phone JPEGs.
PRO_MIN_BLUR = 22.0


@dataclass(frozen=True)
class EditRoute:
    model: str
    use_pro: bool
    reason: str
    scores: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gemini(reason: str, scores: dict[str, Any]) -> EditRoute:
    return EditRoute(
        model=config.RIVERFLOW_MODEL or GEMINI_FLASH_IMAGE,
        use_pro=False,
        reason=reason,
        scores=scores,
    )


def _pro(reason: str, scores: dict[str, Any]) -> EditRoute:
    return EditRoute(
        model=config.RIVERFLOW_PRO_MODEL or RIVERFLOW_PRO,
        use_pro=True,
        reason=reason,
        scores=scores,
    )


def _top_corner_luma(bgr: np.ndarray) -> float:
    h, w = bgr.shape[:2]
    s = max(8, int(0.08 * min(h, w)))
    patches = [bgr[:s, :s], bgr[:s, -s:]]
    pix = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(np.float32)
    return float(pix.mean())


def choose_edit_model(bgr: np.ndarray) -> EditRoute:
    """Pro when the wall is light and the input is sharp."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return _gemini("empty", {})

    scores: dict[str, Any] = {}
    if not config.EDIT_ROUTE_PRO_ON_MESSY_HAIR:
        return _gemini("route_disabled", scores)

    luma = _top_corner_luma(bgr)
    blur = _blur_score(bgr)
    scores = {
        "top_corner_luma": round(luma, 1),
        "blur": round(blur, 2),
        "light_ok": luma >= LIGHT_BG_LUMA_MIN,
        "sharp_ok": blur >= PRO_MIN_BLUR,
    }

    if not scores["light_ok"]:
        return _gemini("bg_not_light", scores)
    if not scores["sharp_ok"]:
        return _gemini("input_not_sharp", scores)
    return _pro("light_sharp", scores)
