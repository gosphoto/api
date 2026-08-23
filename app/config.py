import math
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
# Soft studio / baby phone JPEGs often land ~10–25; keep only trash (<~10) out.
MIN_BLUR_VARIANCE = float(os.getenv("GATE_MIN_BLUR_VARIANCE", "10"))

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "GATE_CORS_ORIGINS",
        "https://gosphoto.ru,https://www.gosphoto.ru,http://localhost:5173",
    ).split(",")
    if o.strip()
]

# EDIT_BACKEND=riverflow|openrouter|local|auto
# EDIT_CUTOUT=silueta|u2netp|u2net|mediapipe|rembg|auto (local fallback)
EDIT_BACKEND = os.getenv("EDIT_BACKEND", "riverflow").strip().lower()
# Always run Gemini/Riverflow edit. Ready studio selfies still go through the model.
SKIP_EDIT_IF_READY = os.getenv("SKIP_EDIT_IF_READY", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
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
OPENROUTER_TIMEOUT_SEC = float(os.getenv("OPENROUTER_TIMEOUT_SEC", "300"))

# Live /api/process edit (OpenRouter /images). Gemini after A/B vs FLUX.2 Pro
# (cheaper, less invented cheek texture on SPUI8374 2026-08-17).
RIVERFLOW_MODEL = os.getenv(
    "RIVERFLOW_MODEL", "google/gemini-2.5-flash-image"
).strip()
# Native white-bg + scoring payload. Used when messy hair sits on a light wall.
RIVERFLOW_PRO_MODEL = os.getenv(
    "RIVERFLOW_PRO_MODEL", "sourceful/riverflow-v2.5-pro"
).strip()
EDIT_ROUTE_PRO_ON_MESSY_HAIR = os.getenv(
    "EDIT_ROUTE_PRO_ON_MESSY_HAIR", "0"
).strip().lower() in ("1", "true", "yes", "on")
RIVERFLOW_BG_MODE = os.getenv("RIVERFLOW_BG_MODE", "solid").strip().lower()
RIVERFLOW_BG_HEX = os.getenv("RIVERFLOW_BG_HEX", "#FFFFFF").strip().strip('"').strip("'")
if RIVERFLOW_BG_HEX and not RIVERFLOW_BG_HEX.startswith("#"):
    RIVERFLOW_BG_HEX = f"#{RIVERFLOW_BG_HEX}"
RIVERFLOW_IMAGE_SIZE = os.getenv("RIVERFLOW_IMAGE_SIZE", "1K").strip()
RIVERFLOW_REASONING = os.getenv("RIVERFLOW_REASONING", "medium").strip().lower()
RIVERFLOW_TIMEOUT_SEC = float(
    os.getenv("RIVERFLOW_TIMEOUT_SEC", str(OPENROUTER_TIMEOUT_SEC))
)

# Passport crop — РФ паспорт, п.34.3 адмрегламента ФМС
# https://rg.ru/documents/2011/08/22/pasport-dok.html
# 35×45 mm @ ≥600 dpi → 827×1063 px; JPEG ≤300 KB
# Загран (Госуслуги): same 35×45 @ ≥300 dpi, larger file budget.
# 300 dpi + round() → 413×531, exactly the Gosuslugi floor — they reject it.
# 360 dpi + ceil() → 497×638, still 35×45 mm, above the pixel/DPI minimum.
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
# Crop aim ~75% (~33.8 mm). Gosuslugi 70–80% = FMS head 32–36 mm.
# Do not hard-fail "oval ≥80%" from §34.3 — it contradicts the 32 mm floor.
PASSPORT_FACE_RATIO = float(os.getenv("PASSPORT_FACE_RATIO", "0.75"))
FACE_RATIO_MIN = float(os.getenv("FACE_RATIO_MIN", "0.70"))
FACE_RATIO_MAX = float(os.getenv("FACE_RATIO_MAX", "0.80"))
# Верхнее поле ~4.5 мм из 45 мм
PASSPORT_TOP_MARGIN = float(os.getenv("PASSPORT_TOP_MARGIN", "0.10"))
HEAD_HEIGHT_MM_MIN = float(os.getenv("HEAD_HEIGHT_MM_MIN", "32"))
HEAD_HEIGHT_MM_MAX = float(os.getenv("HEAD_HEIGHT_MM_MAX", "36"))
HEAD_WIDTH_MM_MIN = float(os.getenv("HEAD_WIDTH_MM_MIN", "18"))
HEAD_WIDTH_MM_MAX = float(os.getenv("HEAD_WIDTH_MM_MAX", "25"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))
JPEG_MAX_BYTES = int(os.getenv("JPEG_MAX_BYTES", str(300 * 1024)))

DOC_TYPE_PASSPORT_RF = "passport_rf"
DOC_TYPE_ZAGRAN = "zagran"
DEFAULT_DOC_TYPE = DOC_TYPE_PASSPORT_RF
ZAGRAN_DPI = int(os.getenv("ZAGRAN_DPI", "360"))
ZAGRAN_JPEG_MAX_BYTES = int(
    os.getenv("ZAGRAN_JPEG_MAX_BYTES", str(2 * 1024 * 1024))
)


def _doc_pixels(dpi: int) -> tuple[int, int]:
    """Pixel size for 35×45 mm at dpi. Ceil so we never undershoot the millimetres."""
    return (
        math.ceil(PASSPORT_WIDTH_MM / 25.4 * dpi),
        math.ceil(PASSPORT_HEIGHT_MM / 25.4 * dpi),
    )


_ZAGRAN_W, _ZAGRAN_H = _doc_pixels(ZAGRAN_DPI)

DOC_PRESETS: dict[str, dict] = {
    DOC_TYPE_PASSPORT_RF: {
        "doc_type": DOC_TYPE_PASSPORT_RF,
        "label": "Паспорт РФ",
        "dpi": PASSPORT_DPI,
        "width": PASSPORT_WIDTH,
        "height": PASSPORT_HEIGHT,
        "jpeg_max_bytes": JPEG_MAX_BYTES,
    },
    DOC_TYPE_ZAGRAN: {
        "doc_type": DOC_TYPE_ZAGRAN,
        "label": "Загранпаспорт",
        "dpi": ZAGRAN_DPI,
        "width": int(os.getenv("ZAGRAN_WIDTH", str(_ZAGRAN_W))),
        "height": int(os.getenv("ZAGRAN_HEIGHT", str(_ZAGRAN_H))),
        "jpeg_max_bytes": ZAGRAN_JPEG_MAX_BYTES,
    },
}


def resolve_doc_preset(doc_type: str | None) -> dict:
    """Return crop/encode preset for passport_rf | zagran (default RF)."""
    key = (doc_type or "").strip().lower() or DEFAULT_DOC_TYPE
    if key not in DOC_PRESETS:
        key = DEFAULT_DOC_TYPE
    return dict(DOC_PRESETS[key])

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

# Feedback form → SMTP (mail.antonbutov.com)
SMTP_HOST = os.getenv("SMTP_HOST", "mail.antonbutov.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "mail@antonbutov.com").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "mail@antonbutov.com").strip()
FEEDBACK_TO = os.getenv("FEEDBACK_TO", "mail@antonbutov.com").strip()
FEEDBACK_RATE_LIMIT = int(os.getenv("FEEDBACK_RATE_LIMIT", "5"))
FEEDBACK_RATE_WINDOW_SEC = int(os.getenv("FEEDBACK_RATE_WINDOW_SEC", "600"))
FEEDBACK_MAX_PHOTO_BYTES = int(
    os.getenv("FEEDBACK_MAX_PHOTO_BYTES", str(5 * 1024 * 1024))
)
FEEDBACK_MAX_MESSAGE_CHARS = int(os.getenv("FEEDBACK_MAX_MESSAGE_CHARS", "4000"))
FEEDBACK_MIN_MESSAGE_CHARS = int(os.getenv("FEEDBACK_MIN_MESSAGE_CHARS", "10"))
# Paid result → customer email (attachments)
RESULT_EMAIL_RATE_LIMIT = int(os.getenv("RESULT_EMAIL_RATE_LIMIT", "5"))
RESULT_EMAIL_RATE_WINDOW_SEC = int(os.getenv("RESULT_EMAIL_RATE_WINDOW_SEC", "600"))

# Tochka acquiring — pay-per-result unlock
TOCHKA_ACCESS_TOKEN = os.getenv("TOCHKA_ACCESS_TOKEN", "").strip()
TOCHKA_MERCHANT_ID = os.getenv("TOCHKA_MERCHANT_ID", "").strip()
TOCHKA_CUSTOMER_CODE = os.getenv("TOCHKA_CUSTOMER_CODE", "").strip()
TOCHKA_API_BASE_URL = os.getenv(
    "TOCHKA_API_BASE_URL", "https://enter.tochka.com"
).rstrip("/")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://gosphoto.ru").rstrip("/")
PRICE_KOPECKS = int(os.getenv("PRICE_KOPECKS", "10000"))
FREE_DOWNLOAD_UNLOCK = os.getenv("FREE_DOWNLOAD_UNLOCK", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
PAYMENT_SYNC_INTERVAL_SECONDS = int(os.getenv("PAYMENT_SYNC_INTERVAL_SECONDS", "30"))
PAYMENTS_DIR = Path(
    os.getenv(
        "PAYMENTS_DIR",
        str(Path(__file__).resolve().parent.parent / "payments"),
    )
)
PAYMENTS_ENABLED = os.getenv("PAYMENTS_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Resume suit upsell (experimental): Pose torso → parallel suit edit → 300 ₽
RESUME_UPSELL_ENABLED = os.getenv("RESUME_UPSELL_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RESUME_PRICE_KOPECKS = int(os.getenv("RESUME_PRICE_KOPECKS", "30000"))
POSE_MODEL_PATH = Path(
    os.getenv(
        "POSE_MODEL_PATH",
        str(
            Path(__file__).resolve().parent.parent
            / "models"
            / "pose_landmarker_lite.task"
        ),
    )
)
# MediaPipe Pose: min landmark visibility for shoulders
TORSO_MIN_VISIBILITY = float(os.getenv("TORSO_MIN_VISIBILITY", "0.45"))
# Mid-shoulder must sit this far below the nose (fraction of image height)
TORSO_MIN_SHOULDER_DROP = float(os.getenv("TORSO_MIN_SHOULDER_DROP", "0.06"))
# Shoulder width as fraction of image width (filters extreme face-crops)
TORSO_MIN_SHOULDER_WIDTH = float(os.getenv("TORSO_MIN_SHOULDER_WIDTH", "0.12"))
