#!/bin/bash
# Instance 1 (React UI) — re-deploy after git push.
# Usage: bash deploy/deploy-react.sh

set -e

APP_DIR="/home/ubuntu/FinanceHQ"
cd "$APP_DIR"

echo "==> Pulling latest code"
git pull origin main

echo "==> Installing React dependencies"
cd react_ui
npm install --prefer-offline --no-audit

echo "==> Building React app"
VITE_API_URL=/api npm run build

cd ..

echo "==> Reloading nginx"
sudo nginx -t && sudo systemctl reload nginx

echo "==> Done — React build deployed"
