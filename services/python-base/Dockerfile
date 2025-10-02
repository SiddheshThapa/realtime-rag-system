FROM python:3.11-slim

RUN addgroup --system app && adduser --system --ingroup app app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git pkg-config libssl-dev \
    wget ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# COPY from repo path (context root) -> /usr/local/bin in image
COPY services/python-base/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER app
ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]
