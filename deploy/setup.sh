#!/bin/bash
# Morpheus — Fresh DigitalOcean Droplet Setup
# Run on a new Ubuntu 24.04 Droplet:
#   ssh root@YOUR_DROPLET_IP 'bash -s' < deploy/setup.sh
set -euo pipefail

REPO_URL="https://github.com/aferna6-cell/Morpheus.git"
BRANCH="merge/neo-integration"
APP_DIR="/opt/morpheus"

echo "=== Morpheus DigitalOcean Setup ==="
echo "Branch: ${BRANCH}"
echo ""

# --- System updates ---
apt-get update -q
apt-get upgrade -y -q
apt-get install -y -q python3 python3-pip python3-venv git ufw fail2ban curl

# --- 2 GB swap (critical for 1 GB RAM plan — 5-model LLM ensemble spikes memory) ---
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    sysctl vm.swappiness=10
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    echo "[ok] 2 GB swap created"
fi

# --- App directory and venv ---
mkdir -p "${APP_DIR}"/{state,logs,runs}

# Clone repo (or update if already cloned)
if [ -d "${APP_DIR}/.git" ]; then
    echo "[ok] Repo exists — pulling latest"
    git -C "${APP_DIR}" fetch origin
    git -C "${APP_DIR}" checkout "${BRANCH}"
    git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"
else
    echo "[ok] Cloning repo"
    git clone --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

# Python venv
if [ ! -d "${APP_DIR}/.venv" ]; then
    python3 -m venv "${APP_DIR}/.venv"
fi
"${APP_DIR}/.venv/bin/pip" install --quiet --upgrade pip
"${APP_DIR}/.venv/bin/pip" install --quiet -r "${APP_DIR}/requirements.txt"
# Extra packages used by kalshi client auth
"${APP_DIR}/.venv/bin/pip" install --quiet cryptography mistralai google-generativeai
echo "[ok] Python dependencies installed"

# --- Systemd services ---
cp "${APP_DIR}/deploy/morpheus.service" /etc/systemd/system/morpheus.service
cp "${APP_DIR}/deploy/morpheus-watchdog.service" /etc/systemd/system/morpheus-watchdog.service
cp "${APP_DIR}/deploy/morpheus-watchdog.timer" /etc/systemd/system/morpheus-watchdog.timer
systemctl daemon-reload
systemctl enable morpheus
systemctl enable morpheus-watchdog.timer
echo "[ok] Systemd services installed"

# --- Log rotation ---
cp "${APP_DIR}/deploy/morpheus-logrotate.conf" /etc/logrotate.d/morpheus
echo "[ok] Log rotation configured"

# --- Firewall: SSH only ---
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw --force enable
echo "[ok] Firewall enabled (SSH only)"

# --- fail2ban: brute-force protection ---
systemctl enable fail2ban
systemctl start fail2ban
echo "[ok] fail2ban enabled"

echo ""
echo "=== NEXT STEP: Create /opt/morpheus/.env ==="
echo ""
echo "cat > /opt/morpheus/.env << 'EOF'"
echo "KALSHI_API_KEY_ID=..."
echo "KALSHI_PRIVATE_KEY_PATH=/opt/morpheus/kalshi_key.pem"
echo "KALSHI_API_KEY_ID_2=..."
echo "KALSHI_PRIVATE_KEY_PATH_2=/opt/morpheus/kalshi_key_2.pem"
echo "OPENAI_API_KEY=..."
echo "ANTHROPIC_API_KEY=..."
echo "MISTRAL_API_KEY=..."
echo "DEEPSEEK_API_KEY=..."
echo "GEMINI_API_KEY=..."
echo "BRAVE_SEARCH_API_KEY=..."
echo "TELEGRAM_BOT_TOKEN=..."
echo "TELEGRAM_CHAT_ID=..."
echo "FRED_API_KEY=..."
echo "EOF"
echo "chmod 600 /opt/morpheus/.env"
echo ""
echo "Also copy your RSA keys:"
echo "  scp kalshi_key.pem kalshi_key_2.pem root@YOUR_IP:/opt/morpheus/"
echo ""
echo "Then start the bot:"
echo "  systemctl start morpheus"
echo "  systemctl start morpheus-watchdog.timer"
echo "  journalctl -u morpheus -f"
