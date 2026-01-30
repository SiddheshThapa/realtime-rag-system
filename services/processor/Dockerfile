FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget git ca-certificates libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY processor.py .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

USER nobody
CMD ["python", "processor.py"]
