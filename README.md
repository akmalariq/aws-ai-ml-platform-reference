# AWS AI/ML Platform Reference

A small but complete reference implementation of an **AWS AI/ML platform**: an
S3 lake house catalogued with Glue and queried with Athena, a training pipeline
with a model registry, a FastAPI inference service, a Streamlit app, and a
retrieval-augmented generation path on Amazon Bedrock.

It exists to make one job description concrete. Every requirement in an
AWS-centric AI/ML engineering role maps to a file here that you can read, run,
and change.

**It runs with no AWS account.** Every AWS call is guarded by `AWSML_USE_AWS`,
so the whole platform executes locally with filesystem and TF-IDF fallbacks and
switches to real AWS services when credentials are present.

## How the JD maps to this repo

| JD area | Where it lives | What it does |
|---|---|---|
| Data science and analytics | `train.py`, `data.py` | Feature engineering, gradient-boosted forecasting, MAE/RMSE/MAPE against a naive baseline |
| ML engineering and MLOps | `pipeline.py`, `train.py`, `registry.py` | Ingest, train, evaluate, register; the SageMaker entry point reads `SM_CHANNEL_TRAIN` and writes `model.tar.gz` |
| AWS lake house | `lakehouse.py`, `sql/athena_ddl.sql`, `infra/terraform` | Parquet in S3, Glue catalog DDL, Athena workgroup |
| AI apps and visualization | `streamlit_app.py`, `serve.py` | Interactive app plus the FastAPI contract it calls |
| Software engineering and deployment | `serve.py`, `Dockerfile`, `.github/workflows/ci.yml`, `infra/terraform` | API, container, CI/CD, Lambda and API Gateway |
| Generative AI and NLP | `rag.py`, `docs/knowledge.md` | Retrieval over a corpus, Amazon Titan and Bedrock on AWS, TF-IDF and extractive locally |
| Business collaboration and strategy | `README.md`, `docs/knowledge.md` | The roadmap, the architecture, and the written rationale |

## Run it locally

```bash
uv sync --all-extras

# ingest, train, evaluate, register
uv run awsml pipeline

# list what is in the lake house and the registry
uv run awsml datasets
uv run awsml models

# ask the corpus a question (local retriever)
uv run awsml rag "How does the model registry work?"

# inference API, then open http://127.0.0.1:8000/docs
uv run awsml serve

# interactive app
uv run --extra app streamlit run src/awsml/streamlit_app.py

# tests
uv run pytest -q
```

The pipeline writes to `output/` by default: `output/lakehouse/` (the lake
house), and `output/models/` (the artifact and `registry.json`).

## Run it on AWS

```bash
export AWSML_USE_AWS=true
export AWSML_BUCKET=<lakehouse-bucket>
export AWS_REGION=ap-southeast-3
# optional, for the Bedrock path
export AWSML_BEDROCK_MODEL_ID=amazon.titan-text-lite-v1

cd infra/terraform
terraform init
terraform plan  -var lakehouse_bucket=<unique> -var artifacts_bucket=<unique>
terraform apply -var lakehouse_bucket=<unique> -var artifacts_bucket=<unique>
```

Then create the Athena tables with `src/awsml/sql/athena_ddl.sql`, replacing
`<bucket>` with `terraform output -raw lakehouse_bucket`. With
`AWSML_USE_AWS=true` the same `awsml pipeline` command writes Parquet to S3, and
`awsml rag` answers through Bedrock.

On AWS the training step becomes a SageMaker Training job calling
`awsml.train.sagemaker_entrypoint`, orchestrated by SageMaker Pipelines or Step
Functions and scheduled by Amazon MWAA. The registry stands in for the SageMaker
Model Registry or MLflow; the interface is the same shape.

## Repository layout

```text
src/awsml/
├── config.py          # the AWSML_USE_AWS switch and settings
├── data.py            # synthetic demand, the stand-in for a Glue ETL output
├── lakehouse.py       # Parquet to S3 or local, the lake house zone
├── train.py           # features, model, metrics, SageMaker entry point
├── registry.py        # versioned model registry
├── pipeline.py        # ingest, train, evaluate, register
├── serve.py           # FastAPI inference API
├── rag.py             # Bedrock or local RAG
├── streamlit_app.py   # interactive app
├── cli.py             # awsml command
└── sql/athena_ddl.sql # Glue and Athena table definitions
infra/terraform/       # S3, Glue, Athena, IAM, Lambda, API Gateway
docs/knowledge.md      # the RAG corpus
tests/
```

## What is real and what is simulated

Honest boundaries, so nothing here overstates:

- **Real and runnable:** the lake house read/write, feature engineering, model
  training and metrics, the registry, the pipeline, the FastAPI service, the
  Streamlit app, the local RAG, and the Terraform.
- **Simulated:** the demand dataset is synthetic, the Lambda handler returns a
  placeholder prediction rather than loading the artifact, and the AWS paths are
  written and validated but have not been executed against a live account here.
- **Not included:** MWAA scheduling, SageMaker Pipelines, and a real Bedrock
  model call are described and wired but need an AWS account and model access to
  run.
- **RAG detail:** the retriever is TF-IDF in both modes; on AWS only the
  generation step switches to Bedrock. Amazon Titan embeddings are described in
  the design but not implemented, so the "embeddings with Titan" pattern is the
  next step, not current behaviour.

## Tests

```bash
uv run pytest
```

Covers the lake house round trip, feature engineering, training metrics, model
save and load, registry versioning, the end-to-end pipeline, RAG retrieval and
the local answer, and the inference API through `TestClient`.
