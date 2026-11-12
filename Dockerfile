# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY configs ./configs
COPY models/detector/weights/best.pt ./models/detector/weights/best.pt
COPY models/classifier/best.pt ./models/classifier/best.pt
COPY tests ./tests
COPY traffic_sign_detector ./traffic_sign_detector
COPY main.py ./main.py

RUN uv sync --frozen --extra dev \
    && mkdir -p data models reports logs

ENTRYPOINT ["uv", "run"]
CMD ["realtime", "--source", "0", "--device", "0", "--log-level", "INFO", "--log-file", "logs/realtime.log"]
