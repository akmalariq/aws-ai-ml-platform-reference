variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-3"
}

variable "project" {
  description = "Name prefix for all resources"
  type        = string
  default     = "awsml"
}

variable "lakehouse_bucket" {
  description = "Globally unique name for the lake house bucket"
  type        = string
}

variable "artifacts_bucket" {
  description = "Globally unique name for the model artifacts and Athena results bucket"
  type        = string
}

variable "glue_database" {
  description = "Glue Data Catalog database name"
  type        = string
  default     = "awsml"
}
