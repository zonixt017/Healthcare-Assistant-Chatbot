# ─────────────────────────────────────────────────────────────────────────────
# Healthcare Assistant Chatbot — Docker Image
# Optimized for HuggingFace Spaces and cloud deployment
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="zonixt017"
LABEL description="Healthcare Assistant Chatbot with RAG"
LABEL version="2.0"

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# HuggingFace Spaces specific
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Create cache directories ───────────────────────────────────────────────────
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers

# ── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY app.py .
COPY data/ ./data/

# ── Create directories for persistence ─────────────────────────────────────────
RUN mkdir -p vectorstore models

# ── Expose port (HF Spaces uses 7860, others use 8501) ─────────────────────────
EXPOSE 7860 8501

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# ── Run Streamlit ──────────────────────────────────────────────────────────────
# Port 7860 for HuggingFace Spaces, can be overridden
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]