"""Inference API.

A FastAPI service that loads the latest registered model and serves predictions.
This is the "APIs using FastAPI" requirement. Deployed on AWS it runs on ECS,
EKS, or Lambda behind API Gateway; the application code is identical.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from awsml.config import Settings
from awsml.registry import latest
from awsml.train import FEATURES, load_model

app = FastAPI(title="awsml inference", version="0.1.0")

_settings = Settings.from_env()
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    entry = latest(_settings.model_name, _settings)
    if entry is None:
        raise HTTPException(status_code=503, detail="no model registered; run the pipeline first")
    path = Path(entry["artifact"])
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"model artifact missing: {path}")
    _model = load_model(path)
    return _model


class PredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(..., description="Engineered feature rows")


class PredictResponse(BaseModel):
    predictions: list[float]
    model_name: str


@app.get("/health")
def health() -> dict:
    entry = latest(_settings.model_name, _settings)
    return {
        "status": "ok",
        "model_name": _settings.model_name,
        "model_version": entry["version"] if entry else None,
        "using_aws": _settings.use_aws,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = _load_model()
    import pandas as pd

    frame = pd.DataFrame(request.rows)
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing}")
    values = model.predict(frame[FEATURES])
    return PredictResponse(
        predictions=[float(value) for value in values], model_name=_settings.model_name
    )
