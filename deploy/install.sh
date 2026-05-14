#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# AI News Pipeline — Automated Install Script
# ──────────────────────────────────────────────

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
INSTALL_DIR="/opt/ai-news-pipeline"
SERVICE_USER="ai-news-pipeline"
VENV_DIR="$INSTALL_DIR/venv"

echo "=== AI News Pipeline Installation ==="

# ── Step 1: Create system user (skip if exists) ──────────────────────────────
echo "[1/8] Creating system user '$SERVICE_USER'..."
if id "$SERVICE_USER" &>/dev/null; then
    echo "  User '$SERVICE_USER' already exists, skipping."
else
    sudo useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    echo "  User '$SERVICE_USER' created."
fi

# ── Step 2: Create directory structure ────────────────────────────────────────
echo "[2/8] Creating directory structure under $INSTALL_DIR..."
sudo mkdir -p "$INSTALL_DIR"/{config,data,logs,src,prompts,tests/fixtures,deploy}

# ── Step 3: Set ownership of install directory ───────────────────────────────
echo "[3/8] Setting ownership to $SERVICE_USER..."
sudo chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"

# ── Step 4: Copy project files ────────────────────────────────────────────────
echo "[4/8] Copying project files..."
sudo cp -r "$PROJECT_DIR/src"    "$INSTALL_DIR/"
sudo cp -r "$PROJECT_DIR/prompts" "$INSTALL_DIR/"
sudo cp -r "$PROJECT_DIR/tests"  "$INSTALL_DIR/"
sudo cp -r "$PROJECT_DIR/config" "$INSTALL_DIR/"
sudo cp "$PROJECT_DIR/requirements.txt" "$INSTALL_DIR/"

# ── Step 5: Create Python virtual environment & install dependencies ─────────
echo "[5/8] Creating Python virtual environment..."
sudo -u "$SERVICE_USER" python3 -m venv "$VENV_DIR"
echo "  Installing dependencies..."
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# ── Step 6: Create .env file from template ────────────────────────────────────
echo "[6/8] Creating .env file from template..."
if [ -f "$INSTALL_DIR/.env" ]; then
    echo "  .env already exists, skipping."
else
    sudo cp "$PROJECT_DIR/.env.example" "$INSTALL_DIR/.env"
    sudo chmod 600 "$INSTALL_DIR/.env"
    sudo chown "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/.env"
    echo "  .env created at $INSTALL_DIR/.env — EDIT THIS FILE with real secrets!"
fi

# ── Step 7: Copy systemd & logrotate configs ─────────────────────────────────
echo "[7/8] Installing systemd units and logrotate config..."
sudo cp "$DEPLOY_DIR/ai-news-pipeline.service" /etc/systemd/system/
sudo cp "$DEPLOY_DIR/ai-news-pipeline.timer"  /etc/systemd/system/
sudo cp "$DEPLOY_DIR/logrotate.conf"           /etc/logrotate.d/ai-news-pipeline

# ── Step 8: Reload systemd, enable & start timer ─────────────────────────────
echo "[8/8] Enabling and starting systemd timer..."
sudo systemctl daemon-reload
sudo systemctl enable ai-news-pipeline.timer
sudo systemctl start  ai-news-pipeline.timer

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation complete ==="
echo ""
echo "Verification commands:"
echo "  sudo systemctl status ai-news-pipeline.timer"
echo "  sudo systemctl status ai-news-pipeline.service"
echo "  sudo journalctl -u ai-news-pipeline.service -n 50 --no-pager"
echo ""
echo "Next steps:"
echo "  1. Edit $INSTALL_DIR/.env and set your API keys"
echo "  2. Review config at $INSTALL_DIR/config/feeds.yaml"
echo "  3. To test the pipeline manually:"
echo "     sudo -u ai-news-pipeline $VENV_DIR/bin/python $INSTALL_DIR/src/main.py --config $INSTALL_DIR/config/feeds.yaml"
