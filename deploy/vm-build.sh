#!/usr/bin/env bash
# Build the Docker image on the demo VM itself and roll it out:
#
#   GABARIT_ADMIN_PASSWORD=<password> bash deploy/vm-build.sh
#
# Packs the committed code (git archive HEAD, a few MB), copies it to the VM,
# builds the image there (no local Docker, no upload of the image from home),
# pushes it to Container Registry and pins the VM spec to the new digest.
# Needs: yc (admin), SSH access to the VM (deploy/yc-setup.sh prints it).
set -euo pipefail
cd "$(dirname "$0")/.."

VM=${VM:-gabarit-demo}
KEY=${SSH_KEY:-$HOME/.ssh/gabarit_demo}
PASS=${GABARIT_ADMIN_PASSWORD:?set GABARIT_ADMIN_PASSWORD}
BASE=ros:humble-ros-base-jammy

REG=$(yc container registry get --name gabarit --format yaml | tr -d '\r' | awk '$1 == "id:" {print $2; exit}')
ADDR=$(yc vpc address get --name gabarit-ip --format yaml | tr -d '\r' | awk '$1 == "address:" {print $2; exit}')
IMAGE=cr.yandex/$REG/gabarit:latest
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "yc-user@$ADDR")
echo ">>> VM $ADDR, image $IMAGE"

# the VM service account pushes the image it builds
yc resource-manager folder add-access-binding "$(yc config get folder-id | tr -d '\r')" \
  --role container-registry.images.pusher \
  --subject "serviceAccount:$(yc iam service-account get --name gabarit-vm --format yaml | tr -d '\r' | awk '$1 == "id:" {print $2; exit}')" \
  >/dev/null 2>&1 || true

TMP=$(mktemp -d)
git archive --format=tar.gz -o "$TMP/src.tar.gz" HEAD
echo ">>> source $(git rev-parse --short HEAD): $(du -h "$TMP/src.tar.gz" | cut -f1)"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$TMP/src.tar.gz" "yc-user@$ADDR:src.tar.gz"
rm -rf "$TMP"

"${SSH[@]}" "BASE='$BASE' IMAGE='$IMAGE' bash -s" <<'REMOTE'
set -euo pipefail
rm -rf ~/gabarit-src && mkdir ~/gabarit-src && tar -xzf ~/src.tar.gz -C ~/gabarit-src
cd ~/gabarit-src
# Docker Hub may be unreachable: fall back to the Google mirror of official images
sudo docker image inspect "$BASE" >/dev/null 2>&1 || sudo docker pull "$BASE" \
  || { sudo docker pull "mirror.gcr.io/library/$BASE" && sudo docker tag "mirror.gcr.io/library/$BASE" "$BASE"; }
sudo docker build -t "$IMAGE" .
TOKEN=$(curl -sf -H Metadata-Flavor:Google \
  http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token \
  | sed -E 's/.*"access_token" *: *"([^"]+)".*/\1/')
echo "$TOKEN" | sudo docker login --username iam --password-stdin cr.yandex >/dev/null
sudo docker push "$IMAGE" | tail -2
sudo docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE" > ~/image-digest
REMOTE

PINNED=$("${SSH[@]}" cat image-digest | tr -d '\r')
echo ">>> built $PINNED"
SPEC=$(mktemp)
sed -e "s#__IMAGE__#$PINNED#" -e "s#__ADMIN_PASSWORD__#$PASS#" deploy/docker-compose.yaml > "$SPEC"
yc compute instance add-metadata --name "$VM" --metadata-from-file docker-compose="$SPEC" >/dev/null
rm -f "$SPEC"
echo ">>> VM spec updated; the container starts in ~1 min: http://$ADDR"
