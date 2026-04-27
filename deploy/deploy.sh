#!/bin/bash
# Re-deploy after pushing new code to GitHub.
# Usage: bash deploy/deploy.sh

set -e

APP_DIR="/home/ubuntu/FinanceHQ"
cd "$APP_DIR"

echo "==> Pulling latest code"
git pull origin main

echo "==> Updating FastAPI dependencies"
venv_api/bin/pip install -r requirements.fastapi.txt --quiet

echo "==> Updating Django dependencies"
venv_ui/bin/pip install -r django_frontend/requirements.django.txt --quiet

echo "==> Collecting static files"
cd django_frontend
../venv_ui/bin/python manage.py collectstatic --noinput --clear
cd ..

echo "==> Restarting services"
sudo systemctl restart financehq-api financehq-ui
sudo systemctl status financehq-api --no-pager
sudo systemctl status financehq-ui --no-pager

echo "==> Done"
