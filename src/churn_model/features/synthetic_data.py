"""Synthetic churn data generator for reproducible testing and demos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_churn_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset with realistic feature signal.
    """
    rng = np.random.default_rng(random_state)

    customer_ids = np.arange(1, n_samples + 1)
    days_since_last_login = rng.exponential(scale=30, size=n_samples).astype(int)
    days_since_last_login = np.clip(days_since_last_login, 0, 365)
    avg_monthly_spend = rng.lognormal(mean=6.5, sigma=1.2, size=n_samples).round(2)
    payment_delays_last_90d = rng.poisson(lam=0.5, size=n_samples)
    support_tickets_last_30d = rng.poisson(lam=1.0, size=n_samples)
    nps_score = rng.integers(low=0, high=11, size=n_samples)
    tenure_months = rng.integers(low=1, high=120, size=n_samples)
    segments = rng.choice(
        ["A", "B", "C", "D", "E"],
        size=n_samples,
        p=[0.35, 0.25, 0.20, 0.15, 0.05],
    )
    contract_type = rng.choice(
        ["monthly", "annual", "biennial"],
        size=n_samples,
        p=[0.4, 0.45, 0.15],
    )

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "days_since_last_login": days_since_last_login,
        "avg_monthly_spend": avg_monthly_spend,
        "payment_delays_last_90d": payment_delays_last_90d,
        "support_tickets_last_30d": support_tickets_last_30d,
        "nps_score": nps_score,
        "tenure_months": tenure_months,
        "segment": segments,
        "contract_type": contract_type,
    })

    # STRONG signal churn score (bigger coefficients, lower noise)
    churn_score = (
        1.5 * (df["days_since_last_login"] > 60).astype(float)
        + 1.2 * (df["payment_delays_last_90d"] >= 2).astype(float)
        + 1.0 * (df["nps_score"] <= 3).astype(float)
        + 0.8 * (df["tenure_months"] < 12).astype(float)
        + 0.6 * (df["support_tickets_last_30d"] >= 3).astype(float)
        + 0.9 * (df["contract_type"] == "monthly").astype(float)
        - 0.5 * (df["avg_monthly_spend"] > 2000).astype(float)
    )
    # Low noise for clean signal
    churn_score = churn_score + rng.normal(0, 0.3, size=n_samples)
    # Sigmoid to probability
    churn_prob = 1 / (1 + np.exp(-1.2 * (churn_score - 1.0)))
    df["churned"] = (rng.uniform(size=n_samples) < churn_prob).astype(int)

    return df


if __name__ == "__main__":
    df = generate_churn_dataset(n_samples=1000)
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(df.head())