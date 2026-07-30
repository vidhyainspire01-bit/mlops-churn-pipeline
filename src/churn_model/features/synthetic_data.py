"""Synthetic churn data generator for reproducible testing and demos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_churn_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset.

    Simulates a business customer base with features that plausibly
    drive churn: engagement, spend patterns, support interactions,
    and payment behavior.

    Args:
        n_samples: number of customer records to generate
        random_state: random seed for reproducibility

    Returns:
        DataFrame with features and binary churn label
    """
    rng = np.random.default_rng(random_state)

    # Customer identifiers
    customer_ids = np.arange(1, n_samples + 1)

    # Engagement features
    days_since_last_login = rng.exponential(scale=30, size=n_samples).astype(int)
    days_since_last_login = np.clip(days_since_last_login, 0, 365)

    # Financial features
    avg_monthly_spend = rng.lognormal(mean=6.5, sigma=1.2, size=n_samples).round(2)
    payment_delays_last_90d = rng.poisson(lam=0.5, size=n_samples)

    # Support / satisfaction features
    support_tickets_last_30d = rng.poisson(lam=1.0, size=n_samples)
    nps_score = rng.integers(low=0, high=11, size=n_samples)

    # Tenure
    tenure_months = rng.integers(low=1, high=120, size=n_samples)

    # Business segment (5 categories to match RAKEZ story)
    segments = rng.choice(
        ["A", "B", "C", "D", "E"],
        size=n_samples,
        p=[0.35, 0.25, 0.20, 0.15, 0.05],
    )

    # Contract type
    contract_type = rng.choice(
        ["monthly", "annual", "biennial"],
        size=n_samples,
        p=[0.4, 0.45, 0.15],
    )

    # Build the DataFrame
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

    # Generate churn label with realistic signal
    # Higher churn risk drivers: low engagement, payment issues, low NPS,
    # short tenure, high support tickets, monthly contracts
    churn_score = (
        0.30 * (df["days_since_last_login"] > 60).astype(float)
        + 0.25 * (df["payment_delays_last_90d"] >= 2).astype(float)
        + 0.20 * (df["nps_score"] <= 3).astype(float)
        + 0.15 * (df["tenure_months"] < 12).astype(float)
        + 0.10 * (df["support_tickets_last_30d"] >= 3).astype(float)
        + 0.15 * (df["contract_type"] == "monthly").astype(float)
        - 0.10 * (df["avg_monthly_spend"] > 2000).astype(float)
    )
    # Add noise so model isn't trivially perfect
    churn_score = churn_score + rng.normal(0, 0.15, size=n_samples)
    # Convert to probability and sample
    churn_prob = 1 / (1 + np.exp(-4 * (churn_score - 0.4)))
    df["churned"] = (rng.uniform(size=n_samples) < churn_prob).astype(int)

    return df


if __name__ == "__main__":
    # Quick smoke test
    df = generate_churn_dataset(n_samples=1000)
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(df.head())