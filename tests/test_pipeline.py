"""Unit tests for the churn pipeline."""

import pandas as pd
import pytest

from src.data_processing import clean_data, split_data
from src.features import add_business_features, build_preprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(200)],
        "tenure_months": [i % 72 for i in range(200)],
        "contract_type": ["Month-to-month", "One year", "Two year"] * 66 + ["Month-to-month", "One year"],
        "monthly_charges": [50.0 + (i % 50) for i in range(200)],
        "total_charges": [500.0 + (i % 500) for i in range(200)],
        "internet_service": ["DSL", "Fiber optic", "No"] * 66 + ["DSL", "Fiber optic"],
        "tech_support": ["Yes", "No"] * 100,
        "online_security": ["Yes", "No"] * 100,
        "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"] * 50,
        "senior_citizen": [0, 1] * 100,
        "partner": ["Yes", "No"] * 100,
        "dependents": ["Yes", "No"] * 100,
        "paperless_billing": ["Yes", "No"] * 100,
        "churn": [0, 0, 0, 1] * 50, 
    })


def test_clean_data_removes_duplicates(sample_df):
    df_with_dupes = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(df_with_dupes)
    assert cleaned["customer_id"].is_unique


def test_clean_data_no_missing_target(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["churn"].isna().sum() == 0


def test_split_data_stratification(sample_df):
    cleaned = clean_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(cleaned, test_size=0.2)

    assert len(X_train) + len(X_test) == len(cleaned)
    assert abs(y_train.mean() - y_test.mean()) < 0.1


def test_add_business_features_creates_columns(sample_df):
    cleaned = clean_data(sample_df)
    enriched = add_business_features(cleaned)

    for col in ["avg_monthly_spend", "spend_deviation", "tenure_bucket", "high_risk_profile"]:
        assert col in enriched.columns


def test_add_business_features_no_division_by_zero(sample_df):
    cleaned = clean_data(sample_df)
    cleaned.loc[0, "tenure_months"] = 0
    enriched = add_business_features(cleaned)
    assert not enriched["avg_monthly_spend"].isna().any()
    assert not enriched["avg_monthly_spend"].isin([float("inf"), float("-inf")]).any()


def test_preprocessor_fits_and_transforms(sample_df):
    cleaned = clean_data(sample_df)
    enriched = add_business_features(cleaned)
    X_train, X_test, y_train, y_test = split_data(enriched)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[1] == X_train_transformed.shape[1]