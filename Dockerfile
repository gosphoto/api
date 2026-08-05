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
      -o /app/models/face_landmarker.task

# Prefetch rembg ONNX weights into image (offline at runtime)
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')"

COPY app ./app

ENV GATE_MODEL_PATH=/app/models/face_landmarker.task \
    EDIT_BACKEND=local \
    REMBG_MODEL=u2net_human_seg \
    PYTHONUNBUFFERED=1

EXPOSE 8091

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
