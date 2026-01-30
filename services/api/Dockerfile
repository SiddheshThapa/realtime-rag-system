FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates wget libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY . .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

USER app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
