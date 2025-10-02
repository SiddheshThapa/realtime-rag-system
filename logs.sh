#!/bin/bash
SERVICE=$1
if [ -z "$SERVICE" ]; then
  echo "Usage: ./logs.sh [service]"
  echo "Available services:"
  docker compose ps --services
  exit 1
fi
docker compose logs -f "$SERVICE"
