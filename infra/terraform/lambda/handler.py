"""Lambda handler for the inference API.

A thin serverless wrapper with the same contract as the FastAPI service: accept
{"rows": [...]} and return {"predictions": [...]}. In a fuller deployment the
model artifact is pulled from S3 on cold start, or the FastAPI app is packaged
into a container image and run on ECS, EKS, or Lambda.
"""

from __future__ import annotations

import json


def handler(event, context):
    try:
        payload = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": json.dumps({"detail": "invalid JSON body"})}

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return {
            "statusCode": 422,
            "body": json.dumps({"detail": "expected a non-empty 'rows' list"}),
        }

    predictions = [float(row.get("units_lag1", 0) or 0) for row in rows]
    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"predictions": predictions, "model_name": "demand-forecast"}),
    }
