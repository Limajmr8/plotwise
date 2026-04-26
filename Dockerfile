FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + model
COPY . .

# Railway sets PORT dynamically
ENV PORT=8080
EXPOSE ${PORT}

CMD uvicorn backend.src.main:app --host 0.0.0.0 --port ${PORT}
