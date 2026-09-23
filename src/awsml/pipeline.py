"""The end-to-end pipeline: ingest, train, evaluate, register.

Locally this is a function you call. On AWS the same steps become a SageMaker
Pipeline (or Step Functions state machine): the ingest step is a Glue job, the
train step is a SageMaker Training job using train.sagemaker_entrypoint, and the
register step is the SageMaker Model Registry.
"""

from __future__ import annotations

from awsml.config import Settings
from awsml.data import generate
from awsml.lakehouse import read_parquet, write_parquet
from awsml.registry import register
from awsml.train import save_model, train


def run(settings: Settings) -> dict:
    """Run the pipeline once and return a summary of what happened."""
    frame = generate()
    lake_location = write_parquet(frame, "demand", settings)

    curated = read_parquet("demand", settings)
    model, metrics = train(curated)

    artifact = settings.path("models", f"{settings.model_name}.joblib")
    save_model(model, artifact)
    entry = register(settings.model_name, str(artifact), metrics, settings)

    return {
        "lakehouse": lake_location,
        "rows": int(len(curated)),
        "metrics": metrics,
        "registry": entry,
    }
