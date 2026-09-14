"""Feature engineering et preprocessor sklearn compilation."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived business variables, which are more informative than the raw data."""
    df = df.copy()

    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
    df["spend_deviation"] = df["monthly_charges"] - df["avg_monthly_spend"]

   
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[-1, 6, 12, 24, 48, 999],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    )

    df["high_risk_profile"] = (
        (df["contract_type"] == "Month-to-month")
        & (df["tech_support"] == "No")
    ).astype(int)

    return df


ENGINEERED_NUMERIC = ["avg_monthly_spend", "spend_deviation"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]
ENGINEERED_BINARY = ["high_risk_profile"]


def build_preprocessor() -> ColumnTransformer:
    """Construit le préprocesseur sklearn (scaling + one-hot encoding)."""
    all_numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC + ENGINEERED_BINARY
    all_categorical = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                all_categorical,
            ),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Retrieves feature names after transformation, for SHAP/interpretation."""
    return list(preprocessor.get_feature_names_out())
