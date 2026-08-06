FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    curl -fsSL \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
      -o /app/models/face_landmarker.task && \
    curl -fsSL \
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite" \
      -o /app/models/selfie_segmenter.tflite && \
    curl -fsSL \
      "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx" \
      -o /app/models/u2netp.onnx

COPY app ./app

ENV GATE_MODEL_PATH=/app/models/face_landmarker.task \
    SELFIE_SEGMENTER_PATH=/app/models/selfie_segmenter.tflite \
    U2NETP_MODEL_PATH=/app/models/u2netp.onnx \
    EDIT_BACKEND=local \
    EDIT_CUTOUT=u2netp \
    PYTHONUNBUFFERED=1

EXPOSE 8091

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
