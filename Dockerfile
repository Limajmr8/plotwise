FROM python:3.11-slim

WORKDIR /app

# 1. Install Python dependencies (rarely changes — cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy ML model + class indices (changes only when retrained — ~31MB)
COPY ml/saved_models/ ml/saved_models/

# 3. Copy data files (rarely changes)
COPY data/sample/ data/sample/

# 4. Copy application code (changes often — narrow layer)
COPY backend/ backend/
COPY frontend/src/ frontend/src/
COPY .env.example .env.example

# Non-root user (required by Hugging Face Spaces; good practice everywhere).
# chown /app so the SQLite DB in data/ stays writable.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Railway/HF set PORT dynamically; HF Spaces reads app_port from README frontmatter
ENV PORT=8080
# Seed ~70 demo disease reports on first boot if the DB is empty (idempotent) —
# keeps the surveillance heatmap populated on ephemeral/fresh containers.
ENV PLOTWISE_SEED_ON_EMPTY=1
EXPOSE ${PORT}

# Healthcheck for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD uvicorn backend.src.main:app --host 0.0.0.0 --port ${PORT}
