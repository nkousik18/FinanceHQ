# Docker and Containerization

## Container Architecture

Three containers, one Docker Compose setup for local development. On Render, each deploys as an independent web service.

```
docker-compose.yml
    │
    ├── fastapi         (RAG backend)
    ├── django          (web frontend)
    └── mlflow          (tracking server)
```

No container for S3 (external). No container for FAISS (in-process). No Redis.

---

## FastAPI Container

### Dockerfile

```dockerfile
# docker/fastapi/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF and FAISS
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.fastapi.txt .
RUN pip install --no-cache-dir -r requirements.fastapi.txt

# Download MiniLM model at build time (not at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY fastapi_service/ .

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key:** MiniLM is downloaded during `docker build`, not at container startup. This prevents a 90MB download on every cold start.

### requirements.fastapi.txt

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
httpx==0.27.0
boto3==1.34.0
pymupdf==1.24.0
sentence-transformers==2.7.0
faiss-cpu==1.8.0
numpy==1.26.0
pydantic-settings==2.2.0
mlflow==2.13.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
nltk==3.8.1
tiktoken==0.7.0
langdetect==1.0.9
python-multipart==0.0.9
```

### Environment Variables (FastAPI)

```bash
# .env.fastapi (never committed to git)

BYTEZ_API_KEY=
BYTEZ_MODEL=mistralai/Mistral-7B-Instruct-v0.3
BYTEZ_MAX_TOKENS=512
BYTEZ_TEMPERATURE=0.1

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=loandoc-bucket

MLFLOW_TRACKING_URI=http://mlflow:5000    # docker-compose internal URL
# On Render: MLFLOW_TRACKING_URI=https://loandoc-mlflow.onrender.com

JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
```

---

## Django Container

### Dockerfile

```dockerfile
# docker/django/Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.django.txt .
RUN pip install --no-cache-dir -r requirements.django.txt

COPY django_frontend/ .

RUN python manage.py collectstatic --noinput

EXPOSE 8080

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8080", "--workers", "2"]
```

### requirements.django.txt

```
django==5.0.0
gunicorn==22.0.0
psycopg2-binary==2.9.9
requests==2.31.0
python-decouple==3.8
whitenoise==6.6.0
```

`whitenoise` serves static files directly from Django — no separate Nginx needed on Render free tier.

### Environment Variables (Django)

```bash
# .env.django

SECRET_KEY=
DEBUG=False
ALLOWED_HOSTS=loandoc.onrender.com,localhost

POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=

FASTAPI_SERVICE_URL=http://fastapi:8000    # docker-compose internal URL
# On Render: FASTAPI_SERVICE_URL=https://loandoc-api.onrender.com
```

---

## MLflow Container

```dockerfile
# docker/mlflow/Dockerfile

FROM python:3.11-slim

RUN pip install mlflow==2.13.0 psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACT_ROOT}", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

### Environment Variables (MLflow)

```bash
MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@host:5432/mlflow_db
MLFLOW_ARTIFACT_ROOT=s3://loandoc-bucket/mlflow/artifacts/
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
```

---

## Docker Compose (Local Development)

```yaml
# docker-compose.yml

version: "3.9"

services:
  fastapi:
    build:
      context: .
      dockerfile: docker/fastapi/Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env.fastapi
    volumes:
      - ./fastapi_service:/app    # hot reload in dev
    command: uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

  django:
    build:
      context: .
      dockerfile: docker/django/Dockerfile
    ports:
      - "8080:8080"
    env_file:
      - .env.django
    environment:
      - FASTAPI_SERVICE_URL=http://fastapi:8000
    depends_on:
      - fastapi
    volumes:
      - ./django_frontend:/app

  mlflow:
    build:
      context: .
      dockerfile: docker/mlflow/Dockerfile
    ports:
      - "5000:5000"
    env_file:
      - .env.mlflow

  # Local Postgres (dev only — use Render Postgres in prod)
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: loandoc
      POSTGRES_USER: loandoc
      POSTGRES_PASSWORD: loandoc
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### Local Dev Commands

```bash
# Start all services
docker compose up --build

# Rebuild a single service
docker compose up --build fastapi

# View logs
docker compose logs -f fastapi

# Run Django migrations
docker compose exec django python manage.py migrate

# Create Django superuser
docker compose exec django python manage.py createsuperuser

# Open FastAPI docs
open http://localhost:8000/docs

# Open MLflow UI
open http://localhost:5000

# Open Django UI
open http://localhost:8080
```

---

## .dockerignore

```
# Root .dockerignore
home/
__pycache__/
*.pyc
*.pyo
.env*
.DS_Store
*.db
*.sqlite3
node_modules/
.git/
venv*/
*.log
test.pdf
```

The `home/` directory (committed venv from old GCP VM) must be excluded.

---

## Image Size Targets

| Container | Target size | Main driver |
|---|---|---|
| FastAPI | < 1.5 GB | MiniLM model (~90MB) + PyMuPDF |
| Django | < 300 MB | Pure Python, no ML |
| MLflow | < 400 MB | Python + MLflow |

Use `python:3.11-slim` (not `python:3.11`) base image to minimize size. Avoid installing dev tools in production images.
