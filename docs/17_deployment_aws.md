# Deployment — AWS EC2 (2-Instance)

**Last updated:** 2026-04-28  
**Status:** Live

---

## Architecture Overview

FinanceHQ runs across two t2.micro EC2 instances (AWS free tier) to stay within the 1GB RAM limit per instance.

| Instance | Role | Services | RAM footprint |
|----------|------|----------|---------------|
| Instance 1 | Django UI | gunicorn + nginx | ~150 MB |
| Instance 2 | FastAPI + MLflow | uvicorn + mlflow + nginx | ~700 MB |

The browser loads the Django UI from Instance 1. JavaScript in the page calls the FastAPI endpoints on Instance 2 directly (configured via `FASTAPI_URL` env var). MLflow runs locally on Instance 2 and is accessed internally by FastAPI.

```
Browser
  │
  ├── GET http://<Instance1> ──► nginx ──► gunicorn:8000 (Django)
  │                                        (renders page with FASTAPI_URL injected)
  │
  └── POST http://<Instance2>/sessions ──► nginx ──► uvicorn:8001 (FastAPI)
      POST http://<Instance2>/query/stream       (SSE streaming, no buffering)
```

---

## Why Two Instances

MiniLM (sentence-transformers) + FastAPI + Django + MLflow all loaded together exceeds 1GB RAM on a single t2.micro, causing OOM crashes. Splitting Django (lightweight) and FastAPI (heavy) across two instances keeps each well within limits.

---

## Instance 1 — Django UI

### EC2 Settings
- AMI: Ubuntu 24.04 LTS
- Type: t2.micro
- Storage: 8 GB
- Security group inbound: SSH (22), HTTP (80) from 0.0.0.0/0

### Services
| Service | Binary | Port |
|---------|--------|------|
| financehq-ui | gunicorn | 127.0.0.1:8000 |
| nginx | nginx | 0.0.0.0:80 |

### Setup
```bash
git clone https://github.com/nkousik18/FinanceHQ.git
cd FinanceHQ
bash deploy/setup-django.sh
cp deploy/env-django.example .env
nano .env   # fill in values
sudo systemctl start financehq-ui nginx
```

### .env (Instance 1)
```
DJANGO_SECRET_KEY=<long random string>
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=<Instance1-public-IP>
FASTAPI_URL=http://<Instance2-public-IP>
LOG_LEVEL=INFO
ENVIRONMENT=production
```

`FASTAPI_URL` is rendered into the Django template and used by the browser to call FastAPI directly — it must be Instance 2's public IP.

---

## Instance 2 — FastAPI + MLflow

### EC2 Settings
- AMI: Ubuntu 24.04 LTS
- Type: t2.micro
- Storage: 8 GB
- Security group inbound: SSH (22), HTTP (80) from 0.0.0.0/0

### Swap (required — prevents OOM during pip install)
Add 1GB swap immediately after launching before running any pip installs:
```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Services
| Service | Binary | Port |
|---------|--------|------|
| financehq-api | uvicorn | 127.0.0.1:8001 |
| financehq-mlflow | mlflow server | 127.0.0.1:5000 |
| nginx | nginx | 0.0.0.0:80 |

### Setup
```bash
# Add swap first (see above)
git clone https://github.com/nkousik18/FinanceHQ.git
cd FinanceHQ
bash deploy/setup-fastapi.sh   # installs CPU-only torch first
cp deploy/env-fastapi.example .env
nano .env   # fill in values
sudo systemctl start financehq-mlflow financehq-api nginx
```

### CPU-only torch
`setup-fastapi.sh` installs PyTorch CPU-only before the rest of the requirements to avoid pulling the 423MB CUDA wheel which exhausts the 8GB disk:
```bash
venv_api/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
venv_api/bin/pip install -r requirements.fastapi.txt
```

### .env (Instance 2)
```
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_REGION=us-east-1
S3_BUCKET=financehq
TEXTRACT_ASYNC_THRESHOLD_PAGES=2
GROQ_API_KEY=<key>
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=1024
GROQ_TEMPERATURE=0.1
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_ARTIFACT_BUCKET=financehq
LOG_LEVEL=INFO
ENVIRONMENT=production
```

---

## nginx Configs

### Instance 1 — `deploy/nginx-django.conf`
Simple proxy to gunicorn. WhiteNoise serves static files so no separate `/static/` block is needed.

### Instance 2 — `deploy/nginx-fastapi.conf`
SSE-safe proxy — buffering disabled so streaming responses reach the browser in real time:
```nginx
proxy_buffering    off;
proxy_cache        off;
proxy_set_header   Connection '';
chunked_transfer_encoding on;
proxy_read_timeout 120s;
```

---

## CI/CD — GitHub Actions

`.github/workflows/deploy.yml` runs on every push to `main`. It SSHes into both instances in parallel and runs the redeploy scripts.

### Required GitHub Secrets
| Secret | Value |
|--------|-------|
| `EC2_SSH_KEY` | Contents of `~/.ssh/ec2-financehq` (private key) |
| `INSTANCE1_IP` | Instance 1 public IP |
| `INSTANCE2_IP` | Instance 2 public IP |

### SSH Key Setup
Generate once on your Mac:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/ec2-financehq -N ""
```

Add the public key to both instances via EC2 Instance Connect browser terminal:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<contents of ec2-financehq.pub>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Add the private key (`cat ~/.ssh/ec2-financehq`) as the `EC2_SSH_KEY` GitHub secret.

### Redeploy Scripts
- `deploy/deploy-django.sh` — git pull, pip install, collectstatic, restart gunicorn
- `deploy/deploy-fastapi.sh` — git pull, pip install, restart uvicorn

---

## IP Change Warning

EC2 public IPs change every time an instance is stopped and started. After a stop/start:

1. Get the new public IP from the AWS console
2. Update `FASTAPI_URL` in Instance 1's `.env` if Instance 2's IP changed
3. Update `INSTANCE1_IP` / `INSTANCE2_IP` GitHub secrets if either IP changed
4. `sudo systemctl restart financehq-ui` on Instance 1 after updating `.env`

To avoid this, assign an **Elastic IP** to each instance (one free per running instance on AWS free tier).

---

## Redeploy After Code Changes

```bash
# Instance 1
cd ~/FinanceHQ && bash deploy/deploy-django.sh

# Instance 2
cd ~/FinanceHQ && bash deploy/deploy-fastapi.sh
```

Or just `git push origin main` — GitHub Actions handles both automatically.

---

## Useful Commands

```bash
# Check service status
sudo systemctl status financehq-ui --no-pager
sudo systemctl status financehq-api --no-pager
sudo systemctl status financehq-mlflow --no-pager

# Tail live logs
sudo journalctl -u financehq-api -f
sudo journalctl -u financehq-ui -f

# Test FastAPI health
curl http://localhost:8001/health       # direct
curl http://localhost/health            # through nginx

# Restart all
sudo systemctl restart financehq-ui nginx            # Instance 1
sudo systemctl restart financehq-api financehq-mlflow nginx  # Instance 2
```
