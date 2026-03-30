#!/bin/bash
# Morpheus — Pull latest code and restart
# Run on the Droplet: bash /opt/morpheus/deploy/update.sh
set -euo pipefail

APP_DIR="/opt/morpheus"
BRANCH="merge/neo-integration"

echo "=== Morpheus Update ==="

# Pull latest
git -C "${APP_DIR}" fetch origin
git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"

# Update dependencies (quiet — only shows changes)
"${APP_DIR}/.venv/bin/pip" install --quiet -r "${APP_DIR}/requirements.txt"

# Reload and restart
systemctl daemon-reload
systemctl restart morpheus

# Wait and show status
sleep 5
echo ""
systemctl status morpheus --no-pager -l
echo ""
echo "Logs: journalctl -u morpheus -f"
