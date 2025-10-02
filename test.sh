#!/bin/bash
set -e
API_URL="http://localhost:8000"

echo "1) Health:"
curl -s ${API_URL}/health || true
echo; echo

echo "2) Get token:"
TOKEN=$(curl -s -X POST ${API_URL}/auth/token | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
echo "TOKEN=$TOKEN"
echo

echo "3) Ready:"
curl -s -H "Auth: Bearer ${TOKEN}" ${API_URL}/ready || true
echo; echo

echo "4) Query:"
curl -s -X POST -H "Auth: Bearer ${TOKEN}" -H "Content-Type: application/json" \
  -d '{"query":"artificial intelligence","k":3}' ${API_URL}/query | sed 's/.*/&\n/' || true
echo

echo "5) Ask:"
curl -s -X POST -H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json" \
  -d '{"question":"summarize the top 5 *association football* wikipedia edits from the last hour","k":8,"max_tokens":120}' \
  ${API_URL}/v1/ask | sed 's/.*/&\n/' || true
echo
