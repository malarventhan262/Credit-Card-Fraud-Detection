
FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
RUN python train_model.py || echo "Training skipped if data not available"

FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app /app

ENV MODEL_DIR=/app/model \
    DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
