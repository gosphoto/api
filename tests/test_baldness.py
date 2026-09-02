import numpy as np

from app.baldness import (
    BALD_GAP_RATIO_MAX,
    classify_gap_ratio,
    person_mask,
    silhouette_top_y,
)
from app.crop import _crop_attempts, face_target_from_gap


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


def test_face_target_from_gap_short_and_full():
    assert face_target_from_gap(0.20) == 0.72
    assert face_target_from_gap(0.30) == 0.72
    assert face_target_from_gap(0.38) == 0.75
    assert face_target_from_gap(0.50) == 0.75
    assert face_target_from_gap(None) == 0.75
    mid = face_target_from_gap(0.34)
    assert 0.72 < mid < 0.75


def test_crop_attempts_bald_prefers_small_crown():
    attempts = _crop_attempts(
        {"is_bald": True, "crown_factor": 0.20, "gap_ratio": 0.20}
    )
    assert attempts[0][0] == 0.20
    assert attempts[0][1] == 0.72
    assert all(a[0] <= 0.35 for a in attempts[:4])


def test_crop_attempts_haired_prefers_silhouette_hint():
    attempts = _crop_attempts(
        {"is_bald": False, "crown_factor": 0.59, "gap_ratio": 0.59}
    )
    assert attempts[0][0] == 0.59
    assert attempts[0][1] == 0.75
    assert attempts[0][2] == 0.10  # PASSPORT_TOP_MARGIN


def test_crop_attempts_haired_default_hint():
    attempts = _crop_attempts({"is_bald": False, "crown_factor": 0.45})
    assert attempts[0][0] == 0.45
