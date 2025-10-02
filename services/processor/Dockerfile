FROM python:3.11-slim

# Faster, more reliable pip in containers
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget git ca-certificates libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy code
COPY services/ ./services/
COPY services/processor/requirements.txt .

# Upgrade base tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only torch + torchvision first
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all remaining requirements (with dependencies!)
RUN pip install --no-cache-dir -r requirements.txt

USER nobody
CMD ["python", "services/processor/processor.py"]
