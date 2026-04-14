FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/emotion_hse/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/emotion_hse/app /app/app
COPY shared /app/shared

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]