#!/usr/bin/env bash
# One-time setup of the cloud demo in Yandex Cloud. Run after `yc init`:
#
#   GABARIT_ADMIN_PASSWORD=<letters and digits> ./deploy/yc-setup.sh
#
# Creates (or reuses): Container Registry "gabarit", service accounts
# "gabarit-ci" (SourceCraft CI: push images, update the VM) and "gabarit-vm"
# (the VM pulls images), a static public IP, a security group (80, 22) and the
# VM "gabarit-demo" on Container Optimized Image. Writes the registry id into
# deploy/ci-cloud.yaml (the deploy workflow, merged into .sourcecraft/ci.yaml
# once the service connection exists). Safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."

ZONE=${ZONE:-ru-central1-a}
VM=${VM:-gabarit-demo}
CORES=${CORES:-4}
MEMORY=${MEMORY:-8}
DISK_GB=${DISK_GB:-100}
PASS=${GABARIT_ADMIN_PASSWORD:?set GABARIT_ADMIN_PASSWORD (letters and digits)}
KEY=${SSH_KEY:-$HOME/.ssh/gabarit_demo}
case "$PASS" in *[!A-Za-z0-9_-]*) echo "password: only letters, digits, _ and -"; exit 1 ;; esac

y() { yc "$@" | tr -d '\r'; }                        # yc.exe on Windows prints CRLF
field() { awk -v k="$1:" '$1 == k {print $2; exit}'; }  # first "key: value" of yaml output

FOLDER=$(yc config get folder-id | tr -d '\r')
[ -n "$FOLDER" ] || { echo "run 'yc init' first"; exit 1; }
echo ">>> folder $FOLDER, zone $ZONE"

[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -f "$KEY" -C gabarit-demo

# --- registry
REG=$(y container registry get --name gabarit --format yaml 2>/dev/null | field id || true)
[ -n "$REG" ] || REG=$(y container registry create --name gabarit --format yaml | field id)
echo ">>> registry $REG"

# --- service accounts and roles
sa() {
  local id
  id=$(y iam service-account get --name "$1" --format yaml 2>/dev/null | field id || true)
  [ -n "$id" ] || id=$(y iam service-account create --name "$1" --format yaml | field id)
  echo "$id"
}
CI_SA=$(sa gabarit-ci)
VM_SA=$(sa gabarit-vm)
bind() { yc resource-manager folder add-access-binding "$FOLDER" --role "$1" --subject "serviceAccount:$2" >/dev/null 2>&1 || true; }
bind container-registry.images.pusher "$CI_SA"
bind compute.editor "$CI_SA"
bind container-registry.images.puller "$VM_SA"
# CI updates a VM that runs as gabarit-vm
yc iam service-account add-access-binding gabarit-vm --role iam.serviceAccounts.user \
  --subject "serviceAccount:$CI_SA" >/dev/null 2>&1 || true
echo ">>> service accounts: gabarit-ci $CI_SA, gabarit-vm $VM_SA"

# --- network: default subnet of the zone, or our own
SUBNET=default-$ZONE
NET=default
if ! yc vpc subnet get --name "$SUBNET" >/dev/null 2>&1; then
  NET=gabarit-net SUBNET=gabarit-$ZONE
  yc vpc network get --name "$NET" >/dev/null 2>&1 || yc vpc network create --name "$NET" >/dev/null
  yc vpc subnet get --name "$SUBNET" >/dev/null 2>&1 || \
    yc vpc subnet create --name "$SUBNET" --zone "$ZONE" --network-name "$NET" --range 10.10.0.0/24 >/dev/null
else
  NET_ID=$(y vpc subnet get --name "$SUBNET" --format yaml | field network_id)
  NET=$(y vpc network get --id "$NET_ID" --format yaml | field name)
fi
SG=$(y vpc security-group get --name gabarit-sg --format yaml 2>/dev/null | field id || true)
[ -n "$SG" ] || SG=$(y vpc security-group create --name gabarit-sg --network-name "$NET" \
  --rule "direction=ingress,port=80,protocol=tcp,v4-cidrs=[0.0.0.0/0]" \
  --rule "direction=ingress,port=22,protocol=tcp,v4-cidrs=[0.0.0.0/0]" \
  --rule "direction=egress,protocol=any,v4-cidrs=[0.0.0.0/0]" --format yaml | field id)
ADDR=$(y vpc address get --name gabarit-ip --format yaml 2>/dev/null | field address || true)
[ -n "$ADDR" ] || ADDR=$(y vpc address create --name gabarit-ip --external-ipv4 zone="$ZONE" --format yaml | field address)
echo ">>> network $NET / $SUBNET, security group $SG, public IP $ADDR"

# --- VM (first start runs :latest; the CI pins the image by digest on every push)
TMP=$(mktemp)
sed -e "s#__IMAGE__#cr.yandex/$REG/gabarit:latest#" -e "s#__ADMIN_PASSWORD__#$PASS#" deploy/docker-compose.yaml > "$TMP"
if yc compute instance get --name "$VM" >/dev/null 2>&1; then
  yc compute instance add-metadata --name "$VM" --metadata-from-file docker-compose="$TMP" >/dev/null
  echo ">>> VM $VM exists, spec updated"
else
  yc compute instance create-with-container --name "$VM" --zone "$ZONE" \
    --cores "$CORES" --memory "$MEMORY" --core-fraction 100 \
    --create-boot-disk size="$DISK_GB",type=network-ssd \
    --network-interface subnet-name="$SUBNET",nat-ip-version=ipv4,nat-address="$ADDR",security-group-ids="$SG" \
    --service-account-name gabarit-vm \
    --ssh-key "$KEY.pub" \
    --docker-compose-file "$TMP" >/dev/null
  echo ">>> VM $VM created"
fi
rm -f "$TMP"

# --- registry id into the deploy workflow
sed -i "s#cr.yandex/__REGISTRY_ID__#cr.yandex/$REG#" deploy/ci-cloud.yaml

cat <<EOF

Done. Next:
  1. SourceCraft -> repository settings -> Service connections: create "gabarit-demo"
     (folder $FOLDER, service account gabarit-ci).
  2. SourceCraft -> repository settings -> Secrets: GABARIT_ADMIN_PASSWORD = the same password.
  3. Add the deploy-demo workflow of deploy/ci-cloud.yaml to .sourcecraft/ci.yaml, commit, git push sourcecraft main
  4. Copy a recording:  ./deploy/upload-bag.sh data/extracted/for_hackathon/doubleT_obstacle
Demo: http://$ADDR   admin: http://$ADDR/admin (admin / your password)
SSH:  ssh -i $KEY yc-user@$ADDR
EOF
