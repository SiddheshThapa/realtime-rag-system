#!/bin/bash
set -e
if [ "${1:-}" = "clean" ]; then
  echo "⚠ Full reset: down + volumes"
  docker compose down -v
  echo "Removing local caches (models/faiss) …"
  rm -rf ./data/faiss/* ./data/models/*
else
  echo "Stopping multi-stage-rag..."
  docker compose down
fi
echo "Stopped."
