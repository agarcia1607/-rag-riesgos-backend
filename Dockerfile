FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para PyMuPDF y sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

ENV RAG_MODE=baseline
ENV PORT=8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
