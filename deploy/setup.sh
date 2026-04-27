#!/bin/bash
# One-time server setup — run once after launching the EC2 instance.
# Usage: bash setup.sh

set -e

REPO="https://github.com/nkousik18/FinanceHQ.git"
APP_DIR="/home/ubuntu/FinanceHQ"

echo "==> Updating system packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip nginx git

# Use whatever python3 is available (3.12 on Ubuntu 24.04)
PYTHON=$(which python3)

echo "==> Cloning repository"
git clone "$REPO" "$APP_DIR"
cd "$APP_DIR"

echo "==> Creating FastAPI venv"
$PYTHON -m venv venv_api
venv_api/bin/pip install --upgrade pip
venv_api/bin/pip install -r requirements.fastapi.txt

echo "==> Creating Django venv"
$PYTHON -m venv venv_ui
venv_ui/bin/pip install --upgrade pip
venv_ui/bin/pip install -r django_frontend/requirements.django.txt

echo "==> Collecting Django static files"
cd django_frontend
../venv_ui/bin/python manage.py collectstatic --noinput
cd ..

echo "==> Installing systemd services"
sudo cp deploy/financehq-api.service    /etc/systemd/system/
sudo cp deploy/financehq-ui.service     /etc/systemd/system/
sudo cp deploy/financehq-mlflow.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable financehq-api financehq-ui financehq-mlflow

echo "==> Configuring nginx"
sudo cp deploy/nginx.conf /etc/nginx/sites-available/financehq
sudo ln -sf /etc/nginx/sites-available/financehq /etc/nginx/sites-enabled/financehq
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx

echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Create your .env file:"
echo "     cp deploy/env.example .env"
echo "     nano .env   (fill in your real values)"
echo ""
echo "  2. Start all services:"
echo "     sudo systemctl start financehq-mlflow financehq-api financehq-ui"
echo "     sudo systemctl start nginx"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status financehq-api financehq-ui"
echo "     sudo journalctl -u financehq-api -f"
echo "============================================================"
