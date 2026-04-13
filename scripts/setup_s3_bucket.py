"""
One-time setup: create S3 bucket with correct settings for FinanceHQ.

Usage:
    python scripts/setup_s3_bucket.py

Reads credentials from .env file.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from app.core.logging import setup_logging, get_logger

load_dotenv()
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), environment="development")
logger = get_logger("setup_s3_bucket")


def create_bucket():
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("AWS_REGION", "us-east-1")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=region,
    )

    logger.info("creating_bucket", bucket=bucket, region=region)

    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        logger.info("bucket_created", bucket=bucket)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            logger.info("bucket_already_exists", bucket=bucket)
        else:
            logger.error("bucket_creation_failed", bucket=bucket, error=str(e))
            sys.exit(1)

    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    logger.info("public_access_blocked", bucket=bucket)

    s3.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    logger.info("versioning_enabled", bucket=bucket)
    logger.info("bucket_ready", uri=f"s3://{bucket}")


if __name__ == "__main__":
    create_bucket()
