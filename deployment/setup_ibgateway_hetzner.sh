#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# setup_ibgateway_hetzner.sh
# Install and configure IB Gateway to run headlessly
# on a Hetzner server via IBC + Xvfb.
# -------------------------------------------------------

# 1. Must run as root
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: This script must be run as root." >&2
  exit 1
fi

# 2. Install Xvfb and required dependencies
echo ">>> Installing dependencies ..."
apt-get update && apt-get install -y xvfb x11vnc wget unzip default-jdk

# 3. Download IB Gateway (stable version)
echo ">>> Downloading IB Gateway installer ..."
wget -O /tmp/ibgateway-installer.sh \
  "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
chmod +x /tmp/ibgateway-installer.sh
echo "Run /tmp/ibgateway-installer.sh manually to install IB Gateway (interactive installer)"

# 4. Download and install IBC (IB Controller)
echo ">>> Installing IBC 3.18.0 ..."
wget -O /tmp/IBC.zip \
  "https://github.com/IbcAlpha/IBC/releases/download/3.18.0/IBCLinux-3.18.0.zip"
mkdir -p /opt/ibc
unzip -o /tmp/IBC.zip -d /opt/ibc
chmod +x /opt/ibc/scripts/*.sh

# 5. Create IBC config
echo ">>> Writing IBC config ..."
cat > /opt/ibc/config.ini <<'IBCCONFIG'
# IBC Configuration — edit credentials before starting the service
FIXLoginId=
FIXPassword=
TradingMode=paper
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
AcceptNonBrokerageAccountWarning=yes
AutoClosedown=no
ExistingSessionDetectedAction=primary
AcceptIncomingConnectionAction=accept
IBCCONFIG

# 6. Create systemd service
echo ">>> Creating systemd service ..."
cat > /etc/systemd/system/ibgateway.service <<'UNIT'
[Unit]
Description=IB Gateway (headless via IBC + Xvfb)
After=network.target

[Service]
Type=simple
User=trading
Environment=DISPLAY=:1
ExecStartPre=/usr/bin/Xvfb :1 -screen 0 1024x768x24 &
ExecStart=/opt/ibc/scripts/ibcstart.sh 1019 --gateway \
  --tws-path=/root/Jts/ibgateway/1019 \
  --ibc-path=/opt/ibc \
  --ibc-ini=/opt/ibc/config.ini
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
UNIT

# 7. Create the trading user if it doesn't exist
id -u trading &>/dev/null || useradd -r -s /bin/false trading

# 8. Final instructions
cat <<'DONE'
============================================================
IB Gateway setup complete.

Next steps:
  1. Run the IB Gateway installer:
     /tmp/ibgateway-installer.sh
  2. Edit credentials:
     nano /opt/ibc/config.ini
  3. Enable and start the service:
     systemctl daemon-reload
     systemctl enable ibgateway
     systemctl start ibgateway
  4. Check status:
     systemctl status ibgateway
     journalctl -u ibgateway -f
============================================================
DONE
