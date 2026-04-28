#!/bin/bash
# Instance 1 (Django UI) — one-time setup.
# Usage: bash deploy/setup-django.sh

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

echo "==> Creating Django venv"
$PYTHON -m venv venv_ui
venv_ui/bin/pip install --upgrade pip
venv_ui/bin/pip install -r django_frontend/requirements.django.txt

echo "==> Collecting Django static files"
cd django_frontend
../venv_ui/bin/python manage.py collectstatic --noinput
cd ..

echo "==> Installing systemd service"
sudo cp deploy/financehq-ui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable financehq-ui

echo "==> Configuring nginx"
sudo cp deploy/nginx-django.conf /etc/nginx/sites-available/financehq
sudo ln -sf /etc/nginx/sites-available/financehq /etc/nginx/sites-enabled/financehq
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx

echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Create your .env file:"
echo "     cp deploy/env-django.example .env"
echo "     nano .env   (fill in FASTAPI_URL with Instance 2 IP)"
echo ""
echo "  2. Start services:"
echo "     sudo systemctl start financehq-ui"
echo "     sudo systemctl start nginx"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status financehq-ui"
echo "     sudo journalctl -u financehq-ui -f"
echo "============================================================"
