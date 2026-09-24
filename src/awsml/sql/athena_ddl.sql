-- Athena DDL for the lake house.
--
-- These statements are the "catalog" step of the AWS lake house: Glue registers
-- the Parquet in S3 as a table, and Athena queries it with SQL, exactly the way
-- the JD describes "unified data access and processing" with Glue and Athena.
--
-- Run them once after `terraform apply` and after the first pipeline run has
-- written Parquet to s3://<bucket>/lakehouse/. Replace <bucket> with the output
-- of `terraform output -raw bucket_name`.

CREATE DATABASE IF NOT EXISTS awsml;

CREATE EXTERNAL TABLE IF NOT EXISTS awsml.demand (
  store INT,
  event_date DATE,
  units INT,
  promo INT
)
STORED AS PARQUET
LOCATION 's3://<bucket>/lakehouse/demand/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');

-- A curated aggregate, the kind of table a BI tool would read.
CREATE TABLE IF NOT EXISTS awsml.demand_daily AS
SELECT
  event_date,
  SUM(units) AS total_units,
  AVG(units) AS avg_units_per_store,
  SUM(promo) AS promo_events
FROM awsml.demand
GROUP BY event_date;
