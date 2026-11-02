"""Soft (non-blocking) gate advisories — no MediaPipe/OpenCV imports."""

from __future__ import annotations

from . import config

WARNING_MESSAGES = {
    "camera_too_close": (
        "Так Госуслуги реже отклоняют фото, и меньше искажений от близкого "
        "фокуса. Держите камеру на расстоянии вытянутой руки, используйте "
        "селфи-палку или попросите другого человека сделать снимок. "
        "Лишнее по краям обрежется автоматически."
    ),
}


def soft_warnings_for_face_height(
    face_height_frac: float,
    *,
    threshold: float | None = None,
) -> tuple[list[str], str | None]:
    """Soft advisory only — never fails the gate."""
    limit = (
        config.WARN_FACE_HEIGHT_FRAC if threshold is None else float(threshold)
    )
    if face_height_frac > limit:
        return (
            ["camera_too_close"],
            WARNING_MESSAGES["camera_too_close"],
        )
    return [], None
