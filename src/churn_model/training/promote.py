"""
Champion/Challenger promotion with schema-aware scoring.
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from churn_model.features.synthetic_data import generate_churn_dataset
from churn_model.training.train import load_config, prepare_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MIN_AUC_IMPROVEMENT = 0.005
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
PREVIOUS_CHAMPION_ALIAS = "previous_champion"


def load_holdout_data(env: str, random_state: int = 999) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a fresh holdout set with a different random state."""
    config = load_config(env)
    df = generate_churn_dataset(
        n_samples=config["data"]["n_samples"] // 2,
        random_state=random_state,
    )
    df = prepare_features(df)
    X = df.drop(columns=["customer_id", "churned"])
    y = df["churned"]
    return X, y


def score_model_by_alias(model_uri: str, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Load a model, try predicting with original int types first.
    If MLflow rejects because it expects float64, retry with cast.
    Handles signature evolution between model versions.
    """
    model = mlflow.pyfunc.load_model(model_uri)

    # Try 1: predict with original int types (works for old int schema)
    try:
        y_proba = model.predict(X)
        logger.info("  Scored with int schema (original)")
    except Exception as e:
        if "float64" in str(e) or "double" in str(e):
            # Model expects float64 - cast and retry
            logger.info("  Model expects float64 schema - casting and retrying")
            X_float = X.astype(
                {c: "float64" for c in X.select_dtypes(include="int").columns}
            )
            y_proba = model.predict(X_float)
        else:
            raise

    if hasattr(y_proba, "shape") and len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]
    return roc_auc_score(y, y_proba)


def promote(env: str) -> None:
    """Compare @challenger vs @champion and promote if better."""
    mlflow.set_registry_uri("databricks-uc")
    config = load_config(env)
    full_model_name = f"{config['catalog']}.{config['schema']}.{config['model_name']}"

    logger.info(f"Promotion check for: {full_model_name} in {env}")

    client = MlflowClient()

    try:
        champion_version = client.get_model_version_by_alias(full_model_name, CHAMPION_ALIAS)
        logger.info(f"@champion is version {champion_version.version}")
    except Exception:
        logger.error("No @champion found - cannot compare. Set a champion first.")
        raise

    try:
        challenger_version = client.get_model_version_by_alias(full_model_name, CHALLENGER_ALIAS)
        logger.info(f"@challenger is version {challenger_version.version}")
    except Exception:
        logger.info("No @challenger found - nothing to promote.")
        return

    if challenger_version.version == champion_version.version:
        logger.info("Challenger is same version as champion. Nothing to do.")
        return

    logger.info("Generating holdout data...")
    X_holdout, y_holdout = load_holdout_data(env)
    logger.info(f"Holdout size: {len(X_holdout)}")

    champion_uri = f"models:/{full_model_name}@{CHAMPION_ALIAS}"
    challenger_uri = f"models:/{full_model_name}@{CHALLENGER_ALIAS}"

    logger.info("Scoring current @champion...")
    champion_auc = score_model_by_alias(champion_uri, X_holdout, y_holdout)
    logger.info(f"@champion AUC: {champion_auc:.4f}")

    logger.info("Scoring @challenger...")
    challenger_auc = score_model_by_alias(challenger_uri, X_holdout, y_holdout)
    logger.info(f"@challenger AUC: {challenger_auc:.4f}")

    improvement = challenger_auc - champion_auc
    logger.info(f"AUC improvement: {improvement:+.4f} (required: {MIN_AUC_IMPROVEMENT:+.4f})")

    if improvement >= MIN_AUC_IMPROVEMENT:
        logger.info("Challenger beats champion - PROMOTING")

        client.set_registered_model_alias(
            name=full_model_name,
            alias=PREVIOUS_CHAMPION_ALIAS,
            version=champion_version.version,
        )
        logger.info(f"  @previous_champion set to version {champion_version.version}")

        client.set_registered_model_alias(
            name=full_model_name,
            alias=CHAMPION_ALIAS,
            version=challenger_version.version,
        )
        logger.info(f"  @champion set to version {challenger_version.version}")

        client.delete_registered_model_alias(name=full_model_name, alias=CHALLENGER_ALIAS)
        logger.info("  @challenger alias removed")

        logger.info(f"PROMOTED version {challenger_version.version} to @champion")
    else:
        logger.info("Challenger does not beat champion by required margin")
        logger.info(f"  Keeping @champion at version {champion_version.version}")
        logger.info(f"  @challenger remains at version {challenger_version.version} for review")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"])
    args = parser.parse_args()
    promote(env=args.env)


if __name__ == "__main__":
    main()
