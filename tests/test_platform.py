"""Tests for the AWS AI/ML platform reference.

Everything runs locally with AWSML_USE_AWS unset, which is the point: the same
code paths that call S3, Glue, SageMaker and Bedrock are exercised with local
fallbacks so the platform is testable with no cloud account.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from awsml.config import Settings
from awsml.data import generate
from awsml.lakehouse import datasets, read_parquet, write_parquet
from awsml.pipeline import run
from awsml.rag import Retriever, answer, load_chunks
from awsml.registry import latest, list_models, register
from awsml.train import FEATURES, build_features, load_model, save_model, train


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        use_aws=False,
        region="ap-southeast-3",
        bedrock_region="ap-southeast-3",
        bucket="awsml-lakehouse-test",
        local_root=tmp_path,
        model_name="demand-forecast",
    )


def test_lakehouse_roundtrip(settings: Settings) -> None:
    frame = generate(seed=1, n_stores=3, days=60)
    location = write_parquet(frame, "demand", settings)
    assert location.endswith("demand.parquet")
    restored = read_parquet("demand", settings)
    assert len(restored) == len(frame)
    assert "demand" in datasets(settings)


def test_feature_engineering() -> None:
    frame = generate(seed=2, n_stores=2, days=60)
    features = build_features(frame)
    for column in FEATURES:
        assert column in features.columns
    assert features["units_lag1"].notna().all()


def test_train_reports_metrics() -> None:
    frame = generate(seed=3, n_stores=4, days=200)
    model, metrics = train(frame, horizon_days=21)
    assert metrics["mae"] > 0
    assert 0 <= metrics["mape"] <= 100
    assert metrics["n_test_rows"] > 0
    assert "improvement_vs_baseline_pct" in metrics


def test_model_roundtrip(settings: Settings, tmp_path: Path) -> None:
    frame = generate(seed=4, n_stores=2, days=80)
    model, _ = train(frame, horizon_days=14)
    path = save_model(model, tmp_path / "model.joblib")
    restored = load_model(path)
    sample = build_features(frame)[FEATURES].head(3)
    assert list(restored.predict(sample)) == list(model.predict(sample))


def test_registry_versions(settings: Settings) -> None:
    first = register("m", "a.joblib", {"mae": 1.0}, settings)
    second = register("m", "b.joblib", {"mae": 0.5}, settings)
    assert first["version"] == 1
    assert second["version"] == 2
    assert latest("m", settings)["artifact"] == "b.joblib"
    assert "m" in list_models(settings)


def test_pipeline_end_to_end(settings: Settings) -> None:
    summary = run(settings)
    assert summary["rows"] > 0
    assert summary["registry"]["version"] == 1
    assert Path(summary["registry"]["artifact"]).exists()


def test_rag_retrieves_and_answers(settings: Settings) -> None:
    chunks = load_chunks()
    assert chunks
    retriever = Retriever(chunks)
    results = retriever.search("How does the model registry work?", top_k=2)
    assert results
    assert results[0][1] > 0
    result = answer("What is the lake house?", settings)
    assert result["backend"] == "local-extractive"
    assert result["sources"]


def test_inference_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWSML_LOCAL_ROOT", str(tmp_path))
    monkeypatch.delenv("AWSML_USE_AWS", raising=False)
    from awsml.config import Settings as Reloaded

    run(Reloaded.from_env())

    from fastapi.testclient import TestClient

    from awsml.serve import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    payload = {
        "rows": [
            {
                "dow": 1,
                "month": 5,
                "is_weekend": 0,
                "promo": 0,
                "units_lag1": 100,
                "units_lag7": 98,
                "units_roll7": 99,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1
