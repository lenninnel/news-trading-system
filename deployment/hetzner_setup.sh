#!/usr/bin/env bash
# Setup FastAPI as a systemd service on Hetzner.
# Run as root: sudo bash deployment/hetzner_setup.sh
set -euo pipefail

SERVICE_NAME="trading-api"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
WORK_DIR="/home/trading/news-trading-system"
ENV_FILE="${WORK_DIR}/.env"
VENV_PYTHON="${WORK_DIR}/.venv/bin/python3"
UVICORN="${WORK_DIR}/.venv/bin/uvicorn"
PORT=8001

echo "[hetzner] Creating systemd service: ${SERVICE_NAME}"

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=News Trading System — FastAPI (read-only API)
After=network.target

[Service]
Type=simple
User=trading
Group=trading
WorkingDirectory=${WORK_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${UVICORN} api.main:app --host 0.0.0.0 --port ${PORT}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "[hetzner] Reloading systemd daemon..."
systemctl daemon-reload

echo "[hetzner] Enabling and starting ${SERVICE_NAME}..."
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

echo "[hetzner] Opening port ${PORT}/tcp via ufw..."
if ufw status | grep -q "${PORT}/tcp"; then
    echo "[hetzner] Port ${PORT}/tcp already open"
else
    ufw allow "${PORT}/tcp"
    echo "[hetzner] Port ${PORT}/tcp opened"
fi

echo "[hetzner] Done. Check status with: systemctl status ${SERVICE_NAME}"
