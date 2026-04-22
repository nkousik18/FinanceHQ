FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.13.0 boto3==1.34.84

WORKDIR /mlflow

EXPOSE 5000

# Shell form so $S3_BUCKET is expanded from the container environment
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://${S3_BUCKET}/mlflow \
    --host 0.0.0.0 \
    --port 5000
