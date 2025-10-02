FROM python:3.11-slim

# Faster, more reliable pip in containers
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates wget libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy code
COPY services/ ./services/
COPY services/api/requirements.txt .

# Upgrade base tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only torch first (avoids heavy CUDA downloads)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all remaining requirements (with dependencies!)
RUN pip install --no-cache-dir -r requirements.txt

USER app
EXPOSE 8000
CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
