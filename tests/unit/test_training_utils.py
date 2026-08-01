"""Unit tests for training utility functions."""
import pandas as pd
import pytest

from churn_model.features.synthetic_data import generate_churn_dataset


def test_generates_requested_number_of_samples():
    df = generate_churn_dataset(n_samples=500)
    assert len(df) == 500


def test_output_has_expected_columns():
    df = generate_churn_dataset(n_samples=100)
    expected_columns = {
        "customer_id",
        "days_since_last_login",
        "avg_monthly_spend",
        "payment_delays_last_90d",
        "support_tickets_last_30d",
        "nps_score",
        "tenure_months",
        "segment",
        "contract_type",
        "churned",
    }
    assert set(df.columns) == expected_columns


def test_churn_label_is_binary():
    df = generate_churn_dataset(n_samples=1000)
    unique_labels = set(df["churned"].unique())
    assert unique_labels.issubset({0, 1})


def test_churn_rate_is_reasonable():
    df = generate_churn_dataset(n_samples=5000)
    churn_rate = df["churned"].mean()
    assert 0.05 < churn_rate < 0.70


def test_days_since_last_login_bounded():
    df = generate_churn_dataset(n_samples=1000)
    assert df["days_since_last_login"].min() >= 0
    assert df["days_since_last_login"].max() <= 365


def test_nps_score_in_valid_range():
    df = generate_churn_dataset(n_samples=1000)
    assert df["nps_score"].min() >= 0
    assert df["nps_score"].max() <= 10


def test_reproducibility_with_same_seed():
    df1 = generate_churn_dataset(n_samples=100, random_state=42)
    df2 = generate_churn_dataset(n_samples=100, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    df1 = generate_churn_dataset(n_samples=100, random_state=42)
    df2 = generate_churn_dataset(n_samples=100, random_state=999)
    assert not df1.equals(df2)


def test_segments_have_expected_values():
    df = generate_churn_dataset(n_samples=1000)
    valid_segments = {"A", "B", "C", "D", "E"}
    assert set(df["segment"].unique()).issubset(valid_segments)


def test_no_null_values_in_features():
    df = generate_churn_dataset(n_samples=1000)
    assert df.isnull().sum().sum() == 0
