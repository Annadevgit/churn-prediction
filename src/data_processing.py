"""Data preparation"""

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "churn"
ID_COL = "customer_id"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "tech_support",
    "online_security",
    "payment_method",
    "senior_citizen",
    "partner",
    "dependents",
    "paperless_billing",
]


def load_data(path: str = "data/raw/churn_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates(subset=ID_COL)

    if df["total_charges"].dtype == object:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print(df.info())
    print(f"\nTaux de churn : {df[TARGET_COL].mean():.2%}")
