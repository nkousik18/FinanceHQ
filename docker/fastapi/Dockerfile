FROM python:3.12-slim

# libgl1 + libglib2.0-0 required by PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.fastapi.txt .
RUN pip install --no-cache-dir -r requirements.fastapi.txt

COPY fastapi_service/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
