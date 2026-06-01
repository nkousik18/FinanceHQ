#!/bin/bash
# Instance 1 (React UI) — one-time setup.
# Usage: bash deploy/setup-react.sh

set -e

REPO="https://github.com/nkousik18/FinanceHQ.git"
APP_DIR="/home/ubuntu/FinanceHQ"

echo "==> Updating system packages"
sudo apt-get update -y
sudo apt-get install -y nginx git curl

echo "==> Installing Node.js 20 LTS"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

echo "==> Cloning repository"
git clone "$REPO" "$APP_DIR"
cd "$APP_DIR"

echo "==> Installing React dependencies"
cd react_ui
npm install --no-audit
cd ..

echo "==> Building React app"
cd react_ui
VITE_API_URL=/api npm run build
cd ..

echo "==> Configuring nginx"
sudo cp deploy/nginx.conf /etc/nginx/sites-available/financehq
sudo ln -sf /etc/nginx/sites-available/financehq /etc/nginx/sites-enabled/financehq
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl start nginx

echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Check the site is live:"
echo "    curl http://localhost/"
echo ""
echo "  Re-deploy after code changes:"
echo "    bash deploy/deploy-react.sh"
echo "============================================================"
