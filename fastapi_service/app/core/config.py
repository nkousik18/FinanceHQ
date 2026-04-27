from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # AWS
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    s3_bucket: str

    # Textract
    textract_async_threshold_pages: int = 2  # pages >= this → async job

    # Groq
    groq_api_key: str = ""
    groq_model: str = "c4ai-aya-expanse-32b"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.1

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_bucket: str = ""

    # App
    log_level: str = "INFO"
    environment: str = "development"

    class Config:
        env_file = (".env", "../.env")   # fastapi_service/.env or project root .env
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
