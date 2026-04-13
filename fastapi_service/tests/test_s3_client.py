"""
Tests for app/storage/s3_client.py

Uses moto to mock AWS S3 — no real AWS calls.
Run:  pytest fastapi_service/tests/test_s3_client.py -v
"""
import json
import pytest
import boto3
from moto import mock_aws
from unittest.mock import patch

from app.storage.s3_client import S3Client, S3Error

BUCKET = "test-loandoc-bucket"
REGION = "us-east-1"


def _make_env_patch():
    """Patch settings so S3Client uses test bucket/region."""
    return patch(
        "app.storage.s3_client.get_settings",
        return_value=type("S", (), {
            "s3_bucket": BUCKET,
            "aws_access_key_id": "testing",
            "aws_secret_access_key": "testing",
            "aws_region": REGION,
        })(),
    )


@mock_aws
class TestS3Client:
    def setup_method(self):
        """Create real mocked S3 bucket before each test."""
        conn = boto3.client(
            "s3",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name=REGION,
        )
        conn.create_bucket(Bucket=BUCKET)

    def _client(self) -> S3Client:
        with _make_env_patch():
            return S3Client()

    # ------------------------------------------------------------------
    # upload_bytes / download_bytes
    # ------------------------------------------------------------------

    def test_upload_and_download_bytes(self):
        client = self._client()
        client.upload_bytes("test/data.bin", b"hello bytes")
        result = client.download_bytes("test/data.bin")
        assert result == b"hello bytes"

    def test_upload_and_download_text(self):
        client = self._client()
        client.upload_text("test/hello.txt", "Hello World")
        result = client.download_text("test/hello.txt")
        assert result == "Hello World"

    def test_upload_json(self):
        client = self._client()
        payload = json.dumps({"key": "value"})
        client.upload_json("test/data.json", payload)
        result = client.download_text("test/data.json")
        assert json.loads(result)["key"] == "value"

    # ------------------------------------------------------------------
    # exists
    # ------------------------------------------------------------------

    def test_exists_returns_true_for_existing_key(self):
        client = self._client()
        client.upload_bytes("test/exists.bin", b"data")
        assert client.exists("test/exists.bin") is True

    def test_exists_returns_false_for_missing_key(self):
        client = self._client()
        assert client.exists("test/nonexistent.bin") is False

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def test_delete_removes_key(self):
        client = self._client()
        client.upload_bytes("test/to_delete.bin", b"data")
        client.delete("test/to_delete.bin")
        assert client.exists("test/to_delete.bin") is False

    # ------------------------------------------------------------------
    # list_keys
    # ------------------------------------------------------------------

    def test_list_keys_returns_matching_prefix(self):
        client = self._client()
        client.upload_bytes("session-1/a.txt", b"a")
        client.upload_bytes("session-1/b.txt", b"b")
        client.upload_bytes("session-2/c.txt", b"c")
        keys = client.list_keys("session-1/")
        assert sorted(keys) == ["session-1/a.txt", "session-1/b.txt"]

    def test_list_keys_returns_empty_for_unknown_prefix(self):
        client = self._client()
        assert client.list_keys("no-such-prefix/") == []

    # ------------------------------------------------------------------
    # get_s3_uri
    # ------------------------------------------------------------------

    def test_get_s3_uri_format(self):
        client = self._client()
        uri = client.get_s3_uri("uploads/sess/original.pdf")
        assert uri == f"s3://{BUCKET}/uploads/sess/original.pdf"

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_download_missing_key_raises_s3_error(self):
        client = self._client()
        with pytest.raises(S3Error):
            client.download_bytes("not/here.txt")
