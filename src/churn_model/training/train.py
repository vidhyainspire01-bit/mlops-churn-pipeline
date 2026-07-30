"""Train a churn prediction model and register to Unity Catalog."""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from churn_model.features.synthetic_data import generate_churn_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(env: str) -> dict:
    """Load environment-specific config from package resources."""
    from importlib.resources import files
    conf_text = files("churn_model.conf").joinpath(f"{env}.yml").read_text()
    return yaml.safe_load(conf_text)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    return pd.get_dummies(df, columns=["segment", "contract_type"], drop_first=True)


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def train(env: str) -> None:
    """
    Full training pipeline: generate data, train, evaluate, register to UC.

    Args:
        env: environment name (dev / staging / prod) — drives config
    """
    config = load_config(env)
    catalog = config["catalog"]
    schema = config["schema"]
    model_name = config["model_name"]
    full_model_name = f"{catalog}.{schema}.{model_name}"

    logger.info(f"Training in environment: {env}")
    logger.info(f"Model will register to: {full_model_name}")

    # Configure MLflow to use Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    experiment_path = f"/Users/thiruadmin82@gmail.com/mlops-churn-pipeline/{env}"
    mlflow.set_experiment(experiment_path)

    # 1. Generate data
    logger.info("Generating training data...")
    df = generate_churn_dataset(
        n_samples=config["data"]["n_samples"],
        random_state=config["training"]["random_state"],
    )
    logger.info(f"Data shape: {df.shape} | Churn rate: {df['churned'].mean():.2%}")

    # 2. Prepare features
    df_features = prepare_features(df)
    X = df_features.drop(columns=["customer_id", "churned"])
    y = df_features["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 3. Train inside MLflow run
    with mlflow.start_run(run_name=f"churn-{env}") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        model_params = {
            "n_estimators": config["training"]["n_estimators"],
            "max_depth": config["training"]["max_depth"],
            "learning_rate": config["training"]["learning_rate"],
            "random_state": config["training"]["random_state"],
            "eval_metric": "logloss",
        }
        mlflow.log_params(model_params)
        mlflow.log_param("environment", env)
        mlflow.log_param("n_samples", config["data"]["n_samples"])

        # Train
        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            logger.info(f"  {name}: {value:.4f}")

        # 4. Quality gate — fail early if model is bad
        gates = config["quality_gates"]
        if metrics["auc"] < gates["min_auc"]:
            raise ValueError(
                f"Quality gate failed: AUC {metrics['auc']:.4f} below threshold {gates['min_auc']}"
            )
        if metrics["precision"] < gates["min_precision"]:
            raise ValueError(
                f"Quality gate failed: Precision {metrics['precision']:.4f} below threshold {gates['min_precision']}"
            )
        logger.info("✓ Quality gates passed")

        # 5. Log model with signature — critical for serving
        signature = infer_signature(X_test, model.predict_proba(X_test))
        input_example = X_test.iloc[:3]

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=full_model_name,
        )
        logger.info(f"✓ Model registered to Unity Catalog: {full_model_name}")

        # 6. Set aliases (first-time or promote)
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(full_model_name)[0].version

        # In dev, always set new version as champion
        # In staging/prod, this would go through challenger flow (Phase 4)
        if env == "dev":
            client.set_registered_model_alias(
                name=full_model_name,
                alias="champion",
                version=latest_version,
            )
            logger.info(f"✓ Version {latest_version} set as @champion in {env}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"])
    args = parser.parse_args()
    train(env=args.env)


if __name__ == "__main__":
    main()