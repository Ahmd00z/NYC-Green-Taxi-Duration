#!/usr/bin/env bash
# Move the canary share to 10, 25, 50 or 100 percent and reload nginx (no downtime).
#   bash deploy/canary_promote.sh 25
set -euo pipefail
PCT="${1:?usage: canary_promote.sh <10|25|50|100>}"
case "$PCT" in 10|25|50|100) ;; *) echo "use 10, 25, 50 or 100"; exit 2;; esac
python -m prodml.canary set "$PCT" --conf deploy/nginx/nginx.conf
docker compose -f docker/docker-compose.canary.yml exec -T nginx nginx -s reload
echo "canary now receives ${PCT}% of traffic"
