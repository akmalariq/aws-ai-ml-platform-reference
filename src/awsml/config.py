"""Configuration, read from the environment so the same code runs locally or on AWS.

The single switch is AWSML_USE_AWS. When it is off (the default), the platform
uses local filesystem paths instead of S3 and skips any AWS API calls. When it
is on, the same functions talk to S3, Glue, Athena, SageMaker and Bedrock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the platform."""

    use_aws: bool
    region: str
    bedrock_region: str
    bucket: str
    local_root: Path
    model_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-southeast-3"))
        return cls(
            use_aws=_env_flag("AWSML_USE_AWS", False),
            region=region,
            bedrock_region=os.getenv("AWSML_BEDROCK_REGION", region),
            bucket=os.getenv("AWSML_BUCKET", "awsml-lakehouse-dev"),
            local_root=Path(os.getenv("AWSML_LOCAL_ROOT", PROJECT_ROOT / "output")),
            model_name=os.getenv("AWSML_MODEL_NAME", "demand-forecast"),
        )

    def path(self, *parts: str) -> Path:
        """A local path under the working root, created on demand."""
        target = self.local_root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def uri(self, *parts: str) -> str:
        """An S3 URI (when use_aws) or a local path string otherwise."""
        joined = "/".join(parts)
        if self.use_aws:
            return f"s3://{self.bucket}/{joined}"
        return str(self.local_root / joined)
