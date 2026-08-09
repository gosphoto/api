import numpy as np

from app.baldness import (
    BALD_GAP_RATIO_MAX,
    classify_gap_ratio,
    person_mask,
    silhouette_top_y,
)
from app.crop import _crop_attempts


def test_classify_bald_gap():
    a = classify_gap_ratio(0.20)
    assert a.is_bald is True
    assert a.crown_factor == 0.20
    assert a.source == "silhouette"


def test_classify_haired_gap():
    a = classify_gap_ratio(0.48)
    assert a.is_bald is False
    assert a.crown_factor == 0.48
    assert a.gap_ratio > BALD_GAP_RATIO_MAX


def test_silhouette_top_finds_head():
    # White canvas with a dark oval whose top is at y=40
    bgr = np.full((200, 120, 3), 255, dtype=np.uint8)
    bgr[40:160, 30:90] = (40, 60, 90)
    top = silhouette_top_y(bgr, mid_x=60.0, half_width=25.0)
    assert top == 40.0
    mask = person_mask(bgr)
    assert bool(mask[40, 60])
    assert not bool(mask[10, 60])


def test_crop_attempts_bald_prefers_small_crown():
    attempts = _crop_attempts(
        {"is_bald": True, "crown_factor": 0.20, "gap_ratio": 0.20}
    )
    assert attempts[0][0] == 0.20
    assert all(a[0] <= 0.35 for a in attempts[:4])


def test_crop_attempts_haired_keeps_defaults():
    attempts = _crop_attempts({"is_bald": False, "crown_factor": 0.45})
    assert attempts[0][0] == 0.45
