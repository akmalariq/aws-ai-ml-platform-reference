# Platform knowledge base

This file is the retrieval corpus for the RAG endpoint. In production the corpus
would be the client's documents; here it is the reference platform's own design.

## Lake house

The lake house stores Parquet in S3 under a `lakehouse/` prefix. AWS Glue crawls
or registers the Parquet as tables in the Glue Data Catalog, and Amazon Athena
queries them with standard SQL. Glue ETL and DataBrew handle ingestion,
transformation, and data preparation before the curated zone is written.

## Training pipeline

Training is a pure Python function so it runs identically in three places: on a
laptop, in CI, and as a managed SageMaker Training job. The SageMaker entry
point reads data from SM_CHANNEL_TRAIN and writes a `model.tar.gz` artifact to
SM_MODEL_DIR. SageMaker Pipelines and Step Functions orchestrate the steps, and
Amazon MWAA (managed Airflow) schedules recurring runs.

## Model registry

Every trained model is registered with a version, its metrics, and the artifact
location. This is the same shape as the SageMaker Model Registry or MLflow: a
pipeline registers a model, and serving always resolves the latest version.

## Inference

The inference service is a FastAPI application. It loads the latest registered
model and exposes a `/predict` endpoint. On AWS it is containerized and runs on
ECS, EKS, or Lambda behind API Gateway. CI/CD builds and deploys it with
CodePipeline, CodeBuild, Terraform, and GitHub Actions.

## Retrieval-augmented generation

The RAG endpoint retrieves the passages most relevant to a question and passes
them to a foundation model as context. On AWS the generation runs through Amazon
Bedrock via the Converse API, for example Amazon Nova through an inference
profile. Retrieval is TF-IDF in both modes; Amazon Titan embeddings are the
intended next step, not current behaviour. The same code falls back to a local
extractive answer when no AWS credentials are present, so the behavior is
testable offline.

## Monitoring and MLOps

Model quality is tracked by MAE, RMSE, and MAPE against a naive baseline, and
each run's metrics are stored with the model version. Drift is detected by
comparing the incoming feature distribution with the training distribution.
