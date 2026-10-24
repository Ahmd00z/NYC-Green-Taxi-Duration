#!/usr/bin/env bash
# Send 100% of traffic back to stable and report how many seconds it took.
set -euo pipefail
START=$(date +%s.%N)
python -m prodml.canary set 0 --conf deploy/nginx/nginx.conf
docker compose -f docker/docker-compose.canary.yml exec -T nginx nginx -s reload
END=$(date +%s.%N)
echo "rollback finished in $(echo "$END - $START" | bc) seconds"
