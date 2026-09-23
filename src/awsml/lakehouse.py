"""The lake house layer.

Writes and reads Parquet in an S3 layout when AWS is enabled, and to a local
directory otherwise. This is the S3 zone of the AWS lake house (Glue catalogs
these files and Athena queries them; the DDL lives in sql/athena_ddl.sql).

Layout, both locally and in S3:

    lakehouse/<dataset>/<dataset>.parquet
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from awsml.config import Settings


def _local_path(dataset: str, settings: Settings) -> Path:
    return settings.path("lakehouse", dataset, f"{dataset}.parquet")


def _s3_key(dataset: str) -> str:
    return f"lakehouse/{dataset}/{dataset}.parquet"


def _s3_client(settings: Settings):
    import boto3

    return boto3.client("s3", region_name=settings.region)


def write_parquet(frame: pd.DataFrame, dataset: str, settings: Settings) -> str:
    """Write a frame to the lake house and return the location written."""
    if settings.use_aws:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        client = _s3_client(settings)
        client.put_object(
            Bucket=settings.bucket, Key=_s3_key(dataset), Body=buffer.getvalue()
        )
        return f"s3://{settings.bucket}/{_s3_key(dataset)}"

    path = _local_path(dataset, settings)
    frame.to_parquet(path, index=False)
    return str(path)


def read_parquet(dataset: str, settings: Settings) -> pd.DataFrame:
    """Read a dataset back from the lake house."""
    if settings.use_aws:
        client = _s3_client(settings)
        response = client.get_object(Bucket=settings.bucket, Key=_s3_key(dataset))
        return pd.read_parquet(io.BytesIO(response["Body"].read()))

    path = _local_path(dataset, settings)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found in the lake house: {dataset}")
    return pd.read_parquet(path)


def datasets(settings: Settings) -> list[str]:
    """List dataset names available in the lake house."""
    if settings.use_aws:
        client = _s3_client(settings)
        prefix = "lakehouse/"
        response = client.list_objects_v2(Bucket=settings.bucket, Prefix=prefix, Delimiter="/")
        return sorted(
            entry["Prefix"].split("/")[-2] for entry in response.get("CommonPrefixes", [])
        )

    root = settings.local_root / "lakehouse"
    if not root.exists():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())
