FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY producer.py .

RUN python -m pip install --upgrade pip \
 && pip install requests sseclient-py kafka-python certifi

CMD ["python","producer.py"]
