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

# EDIT_CUTOUT=silueta|u2netp|u2net|mediapipe|rembg|auto
EDIT_BACKEND = os.getenv("EDIT_BACKEND", "local").strip().lower()
EDIT_CUTOUT = os.getenv("EDIT_CUTOUT", "silueta").strip().lower()
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
U2NETP_MODEL_PATH = Path(
    os.getenv(
        "U2NETP_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "u2netp.onnx"),
    )
)
SILUETA_MODEL_PATH = Path(
    os.getenv(
        "SILUETA_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "silueta.onnx"),
    )
)
U2NET_MODEL_PATH = Path(
    os.getenv(
        "U2NET_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "u2net.onnx"),
    )
)

# OpenRouter edit (optional / fallback)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_IMAGE_MODEL = os.getenv(
    "OPENROUTER_IMAGE_MODEL", "openai/gpt-image-1"
)
# When true, request transparent PNG then composite locally onto white.
OPENROUTER_TRANSPARENT_BG = os.getenv(
    "OPENROUTER_TRANSPARENT_BG", "1"
).strip().lower() in ("1", "true", "yes", "on")
OPENROUTER_TIMEOUT_SEC = float(os.getenv("OPENROUTER_TIMEOUT_SEC", "120"))
# Live site edit: POST /api/process/nano-banana
NANO_BANANA_MODEL = os.getenv(
    "NANO_BANANA_MODEL", "google/gemini-3-pro-image-preview"
).strip()

# Passport crop — РФ паспорт, п.34.3 адмрегламента ФМС
# https://rg.ru/documents/2011/08/22/pasport-dok.html
# 35×45 mm @ ≥600 dpi → 827×1063 px; JPEG ≤300 KB
PASSPORT_WIDTH_MM = 35.0
PASSPORT_HEIGHT_MM = 45.0
PASSPORT_DPI = int(os.getenv("PASSPORT_DPI", "600"))
PASSPORT_WIDTH = int(
    os.getenv(
        "PASSPORT_WIDTH",
        str(round(PASSPORT_WIDTH_MM / 25.4 * PASSPORT_DPI)),
    )
)
PASSPORT_HEIGHT = int(
    os.getenv(
        "PASSPORT_HEIGHT",
        str(round(PASSPORT_HEIGHT_MM / 25.4 * PASSPORT_DPI)),
    )
)
# Овал лица ≥80%; голова в длину 32–36 мм → целевая доля ~0.80 (36/45)
PASSPORT_FACE_RATIO = float(os.getenv("PASSPORT_FACE_RATIO", "0.80"))
# Верхнее поле ~4.5 мм из 45 мм
PASSPORT_TOP_MARGIN = float(os.getenv("PASSPORT_TOP_MARGIN", "0.10"))
HEAD_HEIGHT_MM_MIN = float(os.getenv("HEAD_HEIGHT_MM_MIN", "32"))
HEAD_HEIGHT_MM_MAX = float(os.getenv("HEAD_HEIGHT_MM_MAX", "36"))
HEAD_WIDTH_MM_MIN = float(os.getenv("HEAD_WIDTH_MM_MIN", "18"))
HEAD_WIDTH_MM_MAX = float(os.getenv("HEAD_WIDTH_MM_MAX", "25"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))
JPEG_MAX_BYTES = int(os.getenv("JPEG_MAX_BYTES", str(300 * 1024)))

# Failed gate uploads land here for review (mounted as a volume in compose).
REJECTEDS_DIR = Path(
    os.getenv(
        "REJECTEDS_DIR",
        str(Path(__file__).resolve().parent.parent / "rejecteds"),
    )
)
REJECTEDS_ENABLED = os.getenv("REJECTEDS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Successful /api/process pairs: pairs/<ts>_name/{in.*,out.jpg,meta.json}
PAIRS_DIR = Path(
    os.getenv(
        "PAIRS_DIR",
        str(Path(__file__).resolve().parent.parent / "pairs"),
    )
)
PAIRS_ENABLED = os.getenv("PAIRS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Shareable result pages: results/<32-hex>/{digital.jpg,print.jpg,meta.json}
RESULTS_DIR = Path(
    os.getenv(
        "RESULTS_DIR",
        str(Path(__file__).resolve().parent.parent / "results"),
    )
)
RESULTS_ENABLED = os.getenv("RESULTS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
