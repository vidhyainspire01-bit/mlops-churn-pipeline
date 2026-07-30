# mlops-churn-pipeline
Production-grade MLOps pipeline: Databricks Asset Bundles, Unity Catalog, MLflow aliases, and Champion/Challenger promotion


databricks bundle validate --target dev
databricks bundle deploy --target dev

databricks bundle run churn_training_job --target dev