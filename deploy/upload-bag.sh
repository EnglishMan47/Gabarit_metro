#!/usr/bin/env bash
# Copy a recording (bag directory) to the demo VM and restart the service:
#
#   ./deploy/upload-bag.sh data/extracted/for_hackathon/doubleT_obstacle
#
# Recordings can also be uploaded from the browser ("+ Загрузить").
set -euo pipefail
BAG=${1:?bag directory}
[ -d "$BAG" ] || { echo "not a directory: $BAG"; exit 1; }
KEY=${SSH_KEY:-$HOME/.ssh/gabarit_demo}
ADDR=${ADDR:-$(yc vpc address get --name gabarit-ip --format yaml | tr -d '\r' | awk '$1 == "address:" {print $2; exit}')}
NAME=$(basename "$BAG")
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new yc-user@$ADDR"

echo ">>> $BAG -> $ADDR:/var/lib/gabarit/bags/$NAME"
$SSH "mkdir -p ~/incoming"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new -r "$BAG" "yc-user@$ADDR:~/incoming/"
$SSH "sudo mkdir -p /var/lib/gabarit/bags && sudo rm -rf /var/lib/gabarit/bags/$NAME \
  && sudo mv ~/incoming/$NAME /var/lib/gabarit/bags/ && sudo docker restart gabarit"
echo ">>> done: http://$ADDR"
