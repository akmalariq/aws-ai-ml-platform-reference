output "lakehouse_bucket" {
  description = "S3 bucket holding the lake house Parquet"
  value       = aws_s3_bucket.lakehouse.bucket
}

output "artifacts_bucket" {
  description = "S3 bucket for model artifacts and Athena results"
  value       = aws_s3_bucket.artifacts.bucket
}

output "glue_database" {
  description = "Glue Data Catalog database"
  value       = aws_glue_catalog_database.main.name
}

output "athena_workgroup" {
  description = "Athena workgroup name"
  value       = aws_athena_workgroup.main.name
}

output "inference_api_url" {
  description = "Base URL of the inference API"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "lambda_function_name" {
  description = "Inference Lambda function name"
  value       = aws_lambda_function.inference.function_name
}
