#!/bin/bash
# Instance 2 (FastAPI + MLflow) — one-time setup.
# Usage: bash deploy/setup-fastapi.sh

set -e

REPO="https://github.com/nkousik18/FinanceHQ.git"
APP_DIR="/home/ubuntu/FinanceHQ"

echo "==> Updating system packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip nginx git

PYTHON=$(which python3)

echo "==> Cloning repository"
git clone "$REPO" "$APP_DIR"
cd "$APP_DIR"

echo "==> Creating FastAPI venv (CPU-only torch first to avoid 423MB CUDA wheel)"
$PYTHON -m venv venv_api
venv_api/bin/pip install --upgrade pip
venv_api/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv_api/bin/pip install -r requirements.fastapi.txt

echo "==> Installing systemd services"
sudo cp deploy/financehq-api.service    /etc/systemd/system/
sudo cp deploy/financehq-mlflow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable financehq-api financehq-mlflow

echo "==> Configuring nginx"
sudo cp deploy/nginx-fastapi.conf /etc/nginx/sites-available/financehq
sudo ln -sf /etc/nginx/sites-available/financehq /etc/nginx/sites-enabled/financehq
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx

echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Create your .env file:"
echo "     cp deploy/env-fastapi.example .env"
echo "     nano .env   (fill in AWS keys, Groq key, S3 bucket)"
echo ""
echo "  2. Start services:"
echo "     sudo systemctl start financehq-mlflow"
echo "     sudo systemctl start financehq-api"
echo "     sudo systemctl start nginx"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status financehq-api"
echo "     sudo journalctl -u financehq-api -f"
echo ""
echo "  4. Update Instance 1 .env:"
echo "     Set FASTAPI_URL=http://<this-instance-public-ip>"
echo "     sudo systemctl restart financehq-ui   (on Instance 1)"
echo "============================================================"
