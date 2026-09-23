"""A minimal model registry.

Keeps a JSON index of registered model versions with their metrics and artifact
locations. This is the concept MLflow or the SageMaker Model Registry provides;
the interface is deliberately the same shape (register, list, latest) so it can
be swapped for either without touching the pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from awsml.config import Settings


def _index_path(settings: Settings) -> Path:
    return settings.path("models", "registry.json")


def _read_index(settings: Settings) -> dict:
    path = _index_path(settings)
    if not path.exists():
        return {"models": {}}
    return json.loads(path.read_text())


def _write_index(index: dict, settings: Settings) -> None:
    _index_path(settings).write_text(json.dumps(index, indent=2))


def register(name: str, artifact_path: str, metrics: dict, settings: Settings) -> dict:
    """Register a new version of a model and return the registry entry."""
    index = _read_index(settings)
    versions = index["models"].setdefault(name, [])
    entry = {
        "name": name,
        "version": len(versions) + 1,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact_path,
        "metrics": metrics,
    }
    versions.append(entry)
    _write_index(index, settings)
    return entry


def latest(name: str, settings: Settings) -> dict | None:
    """Return the most recently registered version of a model."""
    versions = _read_index(settings)["models"].get(name, [])
    return versions[-1] if versions else None


def list_models(settings: Settings) -> dict:
    """Return the full registry index."""
    return _read_index(settings)["models"]
