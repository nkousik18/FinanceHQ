#!/bin/bash
# Instance 1 (Django) — re-deploy after git push.
# Usage: bash deploy/deploy-django.sh

set -e

APP_DIR="/home/ubuntu/FinanceHQ"
cd "$APP_DIR"

echo "==> Pulling latest code"
git pull origin main

echo "==> Updating Django dependencies"
venv_ui/bin/pip install -r django_frontend/requirements.django.txt --quiet

echo "==> Collecting static files"
cd django_frontend
../venv_ui/bin/python manage.py collectstatic --noinput --clear
cd ..

echo "==> Restarting Django"
sudo systemctl restart financehq-ui
sudo systemctl status financehq-ui --no-pager

echo "==> Done"
