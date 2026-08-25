FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    curl \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Tochka (enter.tochka.com) serves certs under Russian Trusted Root CA (Минцифры).
# Stock Mozilla/Debian trust store does not include it → SSL CERTIFICATE_VERIFY_FAILED on /pay.
COPY certs/russian_trusted_root_ca.crt \
     certs/russian_trusted_sub_ca.crt \
     /usr/local/share/ca-certificates/
RUN update-ca-certificates

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python - <<'PY'
from pathlib import Path
import certifi
bundle = Path(certifi.where())
extra = Path("/usr/local/share/ca-certificates")
for name in ("russian_trusted_root_ca.crt", "russian_trusted_sub_ca.crt"):
    text = (extra / name).read_text()
    if text.strip() and text not in bundle.read_text():
        with bundle.open("a") as f:
            f.write("\n" + text if not text.endswith("\n") else text)
print("certifi bundle updated:", bundle)
PY

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN mkdir -p /app/models && \
    curl -fsSL \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
      -o /app/models/face_landmarker.task && \
    curl -fsSL \
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" \
      -o /app/models/pose_landmarker_lite.task && \
    curl -fsSL \
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite" \
      -o /app/models/selfie_segmenter.tflite && \
    curl -fsSL \
      "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx" \
      -o /app/models/u2netp.onnx && \
    curl -fsSL \
      "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx" \
      -o /app/models/silueta.onnx

COPY app ./app

ENV GATE_MODEL_PATH=/app/models/face_landmarker.task \
    POSE_MODEL_PATH=/app/models/pose_landmarker_lite.task \
    SELFIE_SEGMENTER_PATH=/app/models/selfie_segmenter.tflite \
    U2NETP_MODEL_PATH=/app/models/u2netp.onnx \
    SILUETA_MODEL_PATH=/app/models/silueta.onnx \
    EDIT_BACKEND=openrouter \
    EDIT_CUTOUT=silueta \
    RESUME_UPSELL_ENABLED=0 \
    RESUME_PRICE_KOPECKS=30000 \
    PYTHONUNBUFFERED=1

EXPOSE 8091

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
