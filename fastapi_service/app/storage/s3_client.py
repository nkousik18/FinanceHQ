"""
S3 client wrapper — single place for all S3 operations.
All methods raise S3Error on failure (never swallow exceptions silently).
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class S3Error(Exception):
    """Raised when any S3 operation fails."""


class S3Client:
    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        logger.info("s3_client_initialized", bucket=self.bucket)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Upload raw bytes to S3 under the given key."""
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            logger.info("s3_upload_ok", key=key, bytes=len(data))
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_upload_failed", key=key, error=str(exc))
            raise S3Error(f"Upload failed for key={key}: {exc}") from exc

    def upload_file(self, key: str, local_path: str) -> None:
        """Upload a local file to S3."""
        try:
            self._client.upload_file(local_path, self.bucket, key)
            logger.info("s3_upload_file_ok", key=key, local_path=local_path)
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_upload_file_failed", key=key, error=str(exc))
            raise S3Error(f"Upload failed for key={key}: {exc}") from exc

    def upload_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        """Upload a UTF-8 string to S3."""
        self.upload_bytes(key, text.encode(encoding), content_type="text/plain")

    def upload_json(self, key: str, data: str) -> None:
        """Upload a JSON string to S3."""
        self.upload_bytes(key, data.encode("utf-8"), content_type="application/json")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_bytes(self, key: str) -> bytes:
        """Download an S3 object and return raw bytes."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            data = response["Body"].read()
            logger.info("s3_download_ok", key=key, bytes=len(data))
            return data
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_download_failed", key=key, error=str(exc))
            raise S3Error(f"Download failed for key={key}: {exc}") from exc

    def download_text(self, key: str, encoding: str = "utf-8") -> str:
        """Download an S3 object and return as string."""
        return self.download_bytes(key).decode(encoding)

    def download_to_file(self, key: str, local_path: str) -> None:
        """Download an S3 object to a local file path."""
        try:
            self._client.download_file(self.bucket, key, local_path)
            logger.info("s3_download_to_file_ok", key=key, local_path=local_path)
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_download_to_file_failed", key=key, error=str(exc))
            raise S3Error(f"Download failed for key={key}: {exc}") from exc

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        """Return True if the key exists in S3."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise S3Error(f"head_object failed for key={key}: {exc}") from exc

    def delete(self, key: str) -> None:
        """Delete a single key from S3."""
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            logger.info("s3_delete_ok", key=key)
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_delete_failed", key=key, error=str(exc))
            raise S3Error(f"Delete failed for key={key}: {exc}") from exc

    def list_keys(self, prefix: str) -> list[str]:
        """List all keys under a prefix. Returns empty list if none found."""
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            logger.info("s3_list_ok", prefix=prefix, count=len(keys))
            return keys
        except (BotoCoreError, ClientError) as exc:
            logger.error("s3_list_failed", prefix=prefix, error=str(exc))
            raise S3Error(f"List failed for prefix={prefix}: {exc}") from exc

    def get_s3_uri(self, key: str) -> str:
        """Return the s3:// URI for a key."""
        return f"s3://{self.bucket}/{key}"


@lru_cache()
def get_s3_client() -> S3Client:
    """Singleton S3 client — reuse across requests."""
    return S3Client()
