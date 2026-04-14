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

    # Bytez
    bytez_api_key: str = ""
    bytez_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    bytez_max_tokens: int = 512
    bytez_temperature: float = 0.1

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_bucket: str = ""

    # App
    log_level: str = "INFO"
    environment: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
