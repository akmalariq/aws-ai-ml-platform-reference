"""Training: the model step of the pipeline.

`train` is the pure function used locally and in tests. `sagemaker_entrypoint`
is the same logic written the way a SageMaker training job calls it, reading
from SM_CHANNEL_TRAIN and writing a model artifact to SM_MODEL_DIR, so the same
code runs as a managed SageMaker job without changes.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "dow",
    "month",
    "is_weekend",
    "promo",
    "units_lag1",
    "units_lag7",
    "units_roll7",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar and lag features from a (store, event_date, units, promo) frame."""
    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out = out.sort_values(["store", "event_date"])
    out["dow"] = out["event_date"].dt.dayofweek
    out["month"] = out["event_date"].dt.month
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    grouped = out.groupby("store", group_keys=False)
    out["units_lag1"] = grouped["units"].shift(1)
    out["units_lag7"] = grouped["units"].shift(7)
    out["units_roll7"] = grouped["units"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=1).mean()
    )
    return out.dropna(subset=["units_lag1", "units_lag7"]).reset_index(drop=True)


def train(frame: pd.DataFrame, horizon_days: int = 28, seed: int = 42) -> tuple[HistGradientBoostingRegressor, dict]:
    """Train on all but the last `horizon_days` per store, evaluate on the holdout."""
    features = build_features(frame)
    cutoff = features["event_date"].max() - pd.Timedelta(days=horizon_days)
    train_set = features[features["event_date"] < cutoff]
    test_set = features[features["event_date"] >= cutoff]

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=6, random_state=seed
    )
    model.fit(train_set[FEATURES], train_set["units"])

    predictions = model.predict(test_set[FEATURES])
    actual = test_set["units"]
    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mape = float(np.mean(np.abs((actual - predictions) / actual.replace(0, np.nan))) * 100)
    naive = test_set["units_lag7"].fillna(actual.median())
    baseline_mae = float(mean_absolute_error(actual, naive))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "baseline_mae": baseline_mae,
        "improvement_vs_baseline_pct": float((1 - mae / baseline_mae) * 100),
        "n_train_rows": int(len(train_set)),
        "n_test_rows": int(len(test_set)),
    }
    return model, metrics


def save_model(model, path: Path) -> Path:
    """Persist a model artifact with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path):
    """Load a model artifact."""
    return joblib.load(path)


def sagemaker_entrypoint() -> None:
    """Entry point for a SageMaker training job.

    SageMaker mounts the training data at SM_CHANNEL_TRAIN and expects the model
    artifact at SM_MODEL_DIR. Nothing here is AWS-specific beyond those paths,
    which is the point: the local and managed paths share the code.
    """
    train_channel = Path(os.environ["SM_CHANNEL_TRAIN"])
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.concat(
        [pd.read_parquet(file) for file in sorted(train_channel.glob("*.parquet"))],
        ignore_index=True,
    )
    model, metrics = train(frame)
    save_model(model, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    archive = model_dir / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(model_dir / "model.joblib", arcname="model.joblib")
    print(json.dumps({"metrics": metrics, "artifact": str(archive)}, indent=2))


if __name__ == "__main__":
    sagemaker_entrypoint()
