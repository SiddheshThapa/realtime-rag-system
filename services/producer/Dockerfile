# services/producer/Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY services/producer/producer.py ./services/producer/producer.py

RUN python -m pip install --upgrade pip \
 && pip install requests sseclient-py kafka-python certifi

CMD ["python","services/producer/producer.py"]
