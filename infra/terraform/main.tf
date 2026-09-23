data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  tags = {
    Project   = var.project
    ManagedBy = "terraform"
  }
}

# ---------------------------------------------------------------------------
# S3: the lake house zone and the artifacts/results zone.
# This is the "AWS lake house for unified data platform" area of the JD.
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "lakehouse" {
  bucket = var.lakehouse_bucket
  tags   = local.tags
}

resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "lakehouse" {
  bucket                  = aws_s3_bucket.lakehouse.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# Glue Data Catalog: the table definitions Athena queries.
# ---------------------------------------------------------------------------

resource "aws_glue_catalog_database" "main" {
  name = var.glue_database
}

# ---------------------------------------------------------------------------
# IAM role for Glue ETL jobs and crawlers.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "glue" {
  name = "${var.project}-glue-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "glue.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3" {
  name = "${var.project}-glue-s3"
  role = aws_iam_role.glue.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.lakehouse.arn,
        "${aws_s3_bucket.lakehouse.arn}/*",
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
    }]
  })
}

# ---------------------------------------------------------------------------
# Athena workgroup: SQL over the lake house.
# ---------------------------------------------------------------------------

resource "aws_athena_workgroup" "main" {
  name = "${var.project}-workgroup"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.artifacts.bucket}/athena-results/"
    }
  }

  tags = local.tags
}

# ---------------------------------------------------------------------------
# Lambda + API Gateway: the inference endpoint.
# On AWS the FastAPI service runs here or on ECS/EKS; this shows the serverless
# path with the same application contract.
# ---------------------------------------------------------------------------

data "archive_file" "inference" {
  type        = "zip"
  source_file = "${path.module}/lambda/handler.py"
  output_path = "${path.module}/.build/inference.zip"
}

resource "aws_iam_role" "lambda" {
  name = "${var.project}-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "inference" {
  function_name    = "${var.project}-inference"
  role             = aws_iam_role.lambda.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.inference.output_path
  source_code_hash = data.archive_file.inference.output_base64sha256

  environment {
    variables = {
      AWSML_BUCKET     = aws_s3_bucket.lakehouse.bucket
      AWSML_MODEL_NAME = "demand-forecast"
      AWSML_USE_AWS    = "true"
    }
  }

  tags = local.tags
}

resource "aws_apigatewayv2_api" "inference" {
  name          = "${var.project}-api"
  protocol_type = "HTTP"
  tags          = local.tags
}

resource "aws_apigatewayv2_integration" "inference" {
  api_id                 = aws_apigatewayv2_api.inference.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.inference.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.inference.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.inference.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.inference.id
  name        = "$default"
  auto_deploy = true
  tags        = local.tags
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.inference.execution_arn}/*/*"
}
