# Deployment — Render

## Why Render

- Free tier supports Docker deployments (unlike Railway's recent changes)
- Free Postgres (1GB)
- Free web services (sleep after 15min inactivity — acceptable for portfolio)
- No credit card required for free tier
- Deploy from GitHub automatically on push

---

## Render Services Setup

Four Render resources are needed:

| Service | Type | Free? | Notes |
|---|---|---|---|
| `loandoc-api` | Web Service (Docker) | Yes | FastAPI RAG backend |
| `loandoc-web` | Web Service (Docker) | Yes | Django frontend |
| `loandoc-mlflow` | Web Service (Docker) | Yes | MLflow tracking server |
| `loandoc-db` | PostgreSQL | Yes | Shared by Django + MLflow |

### Sleep Behavior

Free tier web services sleep after 15 minutes of no traffic. First request after sleep takes ~30 seconds (cold start). For a portfolio demo this is acceptable — warn the reviewer in the README.

To keep MLflow awake (so tracking doesn't fail during queries): use [UptimeRobot](https://uptimerobot.com) (free) to ping `/health` on `loandoc-mlflow` every 10 minutes.

---

## Render Configuration

### loandoc-api (FastAPI)

```yaml
# render.yaml (infrastructure as code)

services:
  - type: web
    name: loandoc-api
    runtime: docker
    dockerfilePath: ./docker/fastapi/Dockerfile
    plan: free
    healthCheckPath: /health
    envVars:
      - key: BYTEZ_API_KEY
        sync: false    # set manually in Render dashboard (secret)
      - key: BYTEZ_MODEL
        value: mistralai/Mistral-7B-Instruct-v0.3
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
      - key: AWS_DEFAULT_REGION
        value: us-east-1
      - key: S3_BUCKET_NAME
        value: loandoc-bucket
      - key: MLFLOW_TRACKING_URI
        fromService:
          name: loandoc-mlflow
          type: web
          property: host
      - key: JWT_SECRET_KEY
        sync: false
```

### loandoc-web (Django)

```yaml
  - type: web
    name: loandoc-web
    runtime: docker
    dockerfilePath: ./docker/django/Dockerfile
    plan: free
    healthCheckPath: /health/
    envVars:
      - key: SECRET_KEY
        sync: false
      - key: DEBUG
        value: "False"
      - key: FASTAPI_SERVICE_URL
        fromService:
          name: loandoc-api
          type: web
          property: host
      - key: DATABASE_URL
        fromDatabase:
          name: loandoc-db
          property: connectionString
```

### loandoc-mlflow

```yaml
  - type: web
    name: loandoc-mlflow
    runtime: docker
    dockerfilePath: ./docker/mlflow/Dockerfile
    plan: free
    healthCheckPath: /health
    envVars:
      - key: MLFLOW_BACKEND_STORE_URI
        fromDatabase:
          name: loandoc-db
          property: connectionString
      - key: MLFLOW_ARTIFACT_ROOT
        value: s3://loandoc-bucket/mlflow/artifacts/
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
      - key: AWS_DEFAULT_REGION
        value: us-east-1

databases:
  - name: loandoc-db
    plan: free
    databaseName: loandoc
```

---

## Deployment Sequence

**First-time setup (order matters):**

1. Create Render Postgres (`loandoc-db`) first — get connection string
2. Deploy `loandoc-mlflow` — wait for it to be healthy, run DB init
3. Deploy `loandoc-api` — set MLFLOW_TRACKING_URI to MLflow service URL
4. Deploy `loandoc-web` — set FASTAPI_SERVICE_URL to FastAPI service URL
5. Run Django migrations: `render run --service loandoc-web python manage.py migrate`
6. Set up UptimeRobot to ping MLflow every 10 minutes

**On every push to `main`:**
- Render auto-deploys all services (if `autoDeploy: yes` in render.yaml)

---

## Environment Variables Security

- All secrets (API keys, JWT secret, AWS credentials) are set in the Render Dashboard UI — never in `render.yaml` or committed to git
- `render.yaml` only contains non-secret config values
- `.env*` files are in `.gitignore`

---

## Health Checks

Every service exposes a `/health` endpoint:

```python
# FastAPI
@app.get("/health")
def health():
    return {"status": "ok", "service": "loandoc-api"}
```

```python
# Django
# urls.py
path("health/", lambda request: JsonResponse({"status": "ok"}))
```

Render uses these for zero-downtime deploys and for detecting crashed containers.

---

## Cold Start Mitigation

Free tier services sleep. To minimize the cold start impact:

1. **FastAPI:** MiniLM model is baked into the Docker image (not downloaded at startup). Cold start is ~15–20 seconds (Python import + Uvicorn startup) instead of ~45 seconds (with model download).

2. **Django:** Static files served via Whitenoise (no separate file server startup needed). Cold start ~10 seconds.

3. **Demo prep:** In the project README, note that "first load may take 30 seconds on free tier." This is standard for portfolio projects on Render.

---

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/deploy.yml

name: Deploy to Render

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run FastAPI tests
        run: |
          pip install -r requirements.fastapi.txt
          pytest fastapi_service/tests/ -v

      - name: Run Django tests
        run: |
          pip install -r requirements.django.txt
          python django_frontend/manage.py test

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Render deploy
        run: |
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_API }}"
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_WEB }}"
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_MLFLOW }}"
```

Tests must pass before Render deploy is triggered. This is the CI/CD gate.

---

## Estimated Monthly Costs

| Resource | Cost |
|---|---|
| Render (3 web services, 1 DB) | $0 |
| AWS S3 | < $0.05 |
| Bytez API (100 queries) | ~$0.10–$0.50 |
| UptimeRobot | $0 |
| **Total** | **< $1/month** |
