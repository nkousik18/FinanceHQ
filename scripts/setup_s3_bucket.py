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

load_dotenv()


def create_bucket():
    bucket = os.environ["S3_BUCKET"]
    region = os.environ.get("AWS_REGION", "us-east-1")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=region,
    )

    print(f"Creating bucket: {bucket} in {region}")

    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"  Bucket created.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  Bucket already exists — skipping.")
        else:
            print(f"  ERROR: {e}")
            sys.exit(1)

    # Block all public access
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print("  Public access blocked.")

    # Enable versioning (cheap safety net)
    s3.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    print("  Versioning enabled.")

    print(f"\nDone. Bucket s3://{bucket} is ready.")


if __name__ == "__main__":
    create_bucket()
