#!/bin/bash
# Deploy Morpheus to DigitalOcean droplet
# Usage: ./deploy.sh <droplet-ip> [--restart]
#
# Prerequisites:
# 1. SSH key access to the droplet
# 2. .env file configured on the droplet at /opt/morpheus/.env
# 3. Python 3.12+ on the droplet

set -euo pipefail

DROPLET_IP="${1:?Usage: ./deploy.sh <droplet-ip> [--restart]}"
RESTART="${2:-}"
REMOTE_DIR="/opt/morpheus"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

echo "==> Deploying Morpheus to ${DROPLET_IP}..."

# 1. Push latest code to GitHub
echo "==> Pushing to GitHub..."
cd "$(dirname "$0")/.."
git add -A
git diff --cached --quiet || git commit -m "Deploy: Morpheus v2 rebuild"
git push origin main 2>/dev/null || git push origin master 2>/dev/null || echo "Push failed or no changes"

# 2. Pull on droplet and set up
echo "==> Pulling code on droplet..."
ssh ${SSH_OPTS} root@${DROPLET_IP} << 'DEPLOY_SCRIPT'
set -euo pipefail

REMOTE_DIR="/opt/morpheus"

# Create directory if needed
mkdir -p ${REMOTE_DIR}

# Clone or pull
if [ -d "${REMOTE_DIR}/.git" ]; then
    cd ${REMOTE_DIR}
    git fetch origin
    git reset --hard origin/main 2>/dev/null || git reset --hard origin/master
else
    git clone git@github.com:aferna6-cell/Morpheus.git ${REMOTE_DIR}
    cd ${REMOTE_DIR}
fi

# Set up venv if needed
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Install dependencies
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt 2>/dev/null || .venv/bin/pip install --quiet -e .

# Create state directory
mkdir -p state runs

# Copy systemd service
cp deploy/systemd/morpheus.service /etc/systemd/system/morpheus.service
systemctl daemon-reload
systemctl enable morpheus

echo "==> Deploy complete on droplet"
DEPLOY_SCRIPT

# 3. Restart if requested
if [ "${RESTART}" = "--restart" ]; then
    echo "==> Restarting Morpheus service..."
    ssh ${SSH_OPTS} root@${DROPLET_IP} "systemctl restart morpheus && sleep 2 && systemctl status morpheus --no-pager"
    echo "==> Service restarted"
else
    echo "==> Code deployed. Run with --restart to restart the service."
    echo "    Or manually: ssh root@${DROPLET_IP} 'systemctl restart morpheus'"
fi

echo "==> Done! Check status: ssh root@${DROPLET_IP} 'journalctl -u morpheus -f'"
