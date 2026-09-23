"""AWS AI/ML platform reference implementation.

A small but complete platform that mirrors the pieces of a production AWS
AI/ML stack: a lake house, a training pipeline with a model registry, an
inference API, an interactive app, and a retrieval-augmented generation path.

Every AWS call is guarded so the whole thing runs locally with no cloud
account, and switches to real AWS services when credentials are present.
"""

__version__ = "0.1.0"
