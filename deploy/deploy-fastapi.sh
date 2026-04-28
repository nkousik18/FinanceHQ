#!/bin/bash
# Instance 2 (FastAPI + MLflow) — re-deploy after git push.
# Usage: bash deploy/deploy-fastapi.sh

set -e

APP_DIR="/home/ubuntu/FinanceHQ"
cd "$APP_DIR"

echo "==> Pulling latest code"
git pull origin main

echo "==> Updating FastAPI dependencies"
venv_api/bin/pip install -r requirements.fastapi.txt --quiet

echo "==> Restarting FastAPI"
sudo systemctl restart financehq-api
sudo systemctl status financehq-api --no-pager

echo "==> Done"
