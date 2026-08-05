import os
from pathlib import Path

MODEL_PATH = Path(
    os.getenv(
        "GATE_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"),
    )
)

MAX_UPLOAD_BYTES = int(os.getenv("GATE_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_IMAGE_SIDE = int(os.getenv("GATE_MAX_IMAGE_SIDE", "1600"))

# Reject if |angle| exceeds these (degrees)
MAX_YAW_DEG = float(os.getenv("GATE_MAX_YAW_DEG", "25"))
MAX_PITCH_DEG = float(os.getenv("GATE_MAX_PITCH_DEG", "25"))
MAX_ROLL_DEG = float(os.getenv("GATE_MAX_ROLL_DEG", "28"))

# Laplacian variance below this → blur
MIN_BLUR_VARIANCE = float(os.getenv("GATE_MIN_BLUR_VARIANCE", "50"))

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "GATE_CORS_ORIGINS",
        "https://gosphoto.ru,https://www.gosphoto.ru,http://localhost:5173",
    ).split(",")
    if o.strip()
]

# Background edit: local MediaPipe cutout by default (free, VPS-safe).
# Set EDIT_BACKEND=openrouter|auto to spend OpenRouter credits.
# EDIT_CUTOUT=mediapipe|rembg — rembg is heavier; only on bigger hosts.
EDIT_BACKEND = os.getenv("EDIT_BACKEND", "local").strip().lower()
EDIT_CUTOUT = os.getenv("EDIT_CUTOUT", "mediapipe").strip().lower()
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2netp").strip()
MIN_PROCESS_SIDE = int(os.getenv("MIN_PROCESS_SIDE", "900"))
SELFIE_SEGMENTER_PATH = Path(
    os.getenv(
        "SELFIE_SEGMENTER_PATH",
        str(
            Path(__file__).resolve().parent.parent
            / "models"
            / "selfie_segmenter.tflite"
        ),
    )
)

# OpenRouter edit (optional / fallback)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_IMAGE_MODEL = os.getenv(
    "OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image"
)
OPENROUTER_TIMEOUT_SEC = float(os.getenv("OPENROUTER_TIMEOUT_SEC", "120"))

# Passport crop output (35×45 mm @ 300 dpi)
PASSPORT_WIDTH = int(os.getenv("PASSPORT_WIDTH", "413"))
PASSPORT_HEIGHT = int(os.getenv("PASSPORT_HEIGHT", "531"))
# RF: face ~70–80% of height; top field ~5 mm of 45 mm ≈ 0.11
PASSPORT_FACE_RATIO = float(os.getenv("PASSPORT_FACE_RATIO", "0.75"))
PASSPORT_TOP_MARGIN = float(os.getenv("PASSPORT_TOP_MARGIN", "0.11"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "95"))
