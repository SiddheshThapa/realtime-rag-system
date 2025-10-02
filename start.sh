#!/bin/bash
set -euo pipefail

echo "▶ multi-stage-rag — starting..."

# 0) .env bootstrap
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "✔ Created .env from .env.example (edit if you want to change secrets or models)."
  else
    echo "❌ .env missing and .env.example not found. Aborting."
    exit 1
  fi
fi

# Load env values we care about (safe; ignores comments)
export OLLAMA_URL="$(grep -E '^OLLAMA_URL=' .env | cut -d= -f2- || true)"
export OLLAMA_MODEL="$(grep -E '^OLLAMA_MODEL=' .env | cut -d= -f2- || true)"

# 1) Folders that containers share
mkdir -p data/faiss data/models infra/postgres-init

# 2) Preload HF weights to host cache so api/processor don’t redownload on each rebuild
echo "▶ Preloading HF models (safe to skip if already present)..."
docker compose build models-preload
docker compose run --rm models-preload || true

# 3) Bring up base infra + (optional) ollama first, so api/processor can find them
echo "▶ Bringing up core services..."
docker compose up -d zookeeper kafka postgres redis jaeger prometheus grafana || true

# 4) If you ship an ollama service in compose, start it now
if docker compose ps --services | grep -q '^ollama$'; then
  echo "▶ Starting ollama..."
  docker compose up -d ollama
fi

# 5) Build & start app services
echo "▶ Building app images (api/processor/producer)..."
docker compose up -d --build processor api producer || true

# 6) If OLLAMA_URL points to our compose service (http://ollama:11434), pull the model inside the container
if [[ "${OLLAMA_URL:-}" =~ ^http://ollama:11434/?$ ]] && [ -n "${OLLAMA_MODEL:-}" ]; then
  echo "▶ Pulling Ollama model '${OLLAMA_MODEL}' inside the container..."
  # Wait a bit in case the daemon is still booting
  sleep 2
  docker compose exec -T ollama bash -lc "ollama pull ${OLLAMA_MODEL}" || true
fi

# 7) Wait for API health
API_URL="http://localhost:8000"
echo "▶ Waiting for API to be healthy at ${API_URL}/health ..."
for i in {1..40}; do
  if curl -fsS "${API_URL}/health" >/dev/null 2>&1; then
    echo "✔ API is up."
    break
  fi
  sleep 1
  if [ $i -eq 40 ]; then
    echo "❌ API did not become healthy in time. Try './logs.sh api'."
    exit 1
  fi
done

# 8) If no vectors yet, seed with the sample producer so queries return something even if SSE is blocked
echo "▶ Checking vector store count..."
if docker compose exec -T redis redis-cli scard vectors:ids | grep -qE '^\(integer\) 0$'; then
  echo "ℹ No vectors yet — running sample producer once..."
  docker compose run --rm producer_sample || true
else
  echo "✔ Vector store already has data."
fi

echo
echo "🎉 Ready!"
echo "• Swagger:   ${API_URL}/docs"
echo "• Health:    ${API_URL}/health"
echo "• Readiness: ${API_URL}/ready"
echo "• Follow logs: ./logs.sh api   (or producer / processor)"
