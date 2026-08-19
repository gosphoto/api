"""Pick Riverflow Pro vs Gemini from cheap CPU cues on the input selfie."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from . import config
from .gate import _blur_score

GEMINI_FLASH_IMAGE = "google/gemini-2.5-flash-image"
RIVERFLOW_PRO = "sourceful/riverflow-v2.5-pro"

# Top-corner wall: her selfie ~231; dark rooms sit well below this.
LIGHT_BG_LUMA_MIN = 185.0
# Gate floor is 10; Pro is wasted on mushy phone JPEGs.
PRO_MIN_BLUR = 22.0
HAIR_DARK_MAX = 125
HAIR_Y_FRAC = 0.50
WISP_FRAC_MIN = 0.05
HAIR_PIX_MIN = 180


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


def _hair_wisp_frac(bgr: np.ndarray) -> tuple[float, int]:
    """Fraction of upper dark mass that is thin (flyaways), vs a solid hair blob.

    Always measured on a ~400px canvas so a 12MP selfie and a 400px fixture
    share the same threshold.
    """
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side > 400:
        s = 400.0 / side
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(w * s))), max(1, int(round(h * s)))),
            interpolation=cv2.INTER_AREA,
        )
        h, w = bgr.shape[:2]
    y_cut = max(8, int(HAIR_Y_FRAC * h))
    gray = cv2.cvtColor(bgr[:y_cut], cv2.COLOR_BGR2GRAY)
    hair = (gray < HAIR_DARK_MAX).astype(np.uint8) * 255
    n = int((hair > 0).sum())
    if n < HAIR_PIX_MIN:
        return 0.0, n
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(hair, cv2.MORPH_OPEN, k)
    wisps = (hair > 0) & (opened == 0)
    return float(wisps.sum()) / float(n), n


def choose_edit_model(bgr: np.ndarray) -> EditRoute:
    """Pro only when hair is messy, the wall is light, and the input is sharp."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return _gemini("empty", {})

    scores: dict[str, Any] = {}
    if not config.EDIT_ROUTE_PRO_ON_MESSY_HAIR:
        return _gemini("route_disabled", scores)

    luma = _top_corner_luma(bgr)
    blur = _blur_score(bgr)
    wisp_frac, hair_pix = _hair_wisp_frac(bgr)
    scores = {
        "top_corner_luma": round(luma, 1),
        "blur": round(blur, 2),
        "wisp_frac": round(wisp_frac, 3),
        "hair_pix": hair_pix,
        "light_ok": luma >= LIGHT_BG_LUMA_MIN,
        "sharp_ok": blur >= PRO_MIN_BLUR,
        "messy_ok": wisp_frac >= WISP_FRAC_MIN and hair_pix >= HAIR_PIX_MIN,
    }

    if not scores["light_ok"]:
        return _gemini("bg_not_light", scores)
    if not scores["sharp_ok"]:
        return _gemini("input_not_sharp", scores)
    if not scores["messy_ok"]:
        return _gemini("hair_neat", scores)
    return _pro("messy_hair_light_sharp", scores)
