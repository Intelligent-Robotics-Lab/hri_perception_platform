FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/affect_hse_va/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/affect_hse_va/app /app/app
COPY shared /app/shared

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]