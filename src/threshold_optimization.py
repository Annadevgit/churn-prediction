"""
Decision threshold optimization via cost-benefit analysis.

The default threshold of 0.5 has no business justification. This script finds
the threshold that minimizes the real total cost, given:
- the cost of a false positive (contacting a customer who wasn't going to leave)
- the cost of a false negative (losing a customer without having tried to retain them)
From what probability should we consider that a customer is going to churn and act on it
"""


import joblib
import numpy as np
import pandas as pd

from src.data_processing import clean_data, load_data, split_data
from src.features import add_business_features

# Business assumptions — adjust with real figures if available
COST_FALSE_POSITIVE = 5      # cost of an unnecessary retention action (e.g. discount offered)
COST_FALSE_NEGATIVE = 500    # lost customer lifetime value if churn is not anticipated
COST_TRUE_POSITIVE = 50      # cost of a successful retention action (discount + sales effort)


def compute_cost(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total_cost = (
        tp * COST_TRUE_POSITIVE
        + fp * COST_FALSE_POSITIVE
        + fn * COST_FALSE_NEGATIVE
    )

    return {
        "threshold": round(threshold, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_cost": total_cost,
        "customers_contacted": tp + fp,
    }


def find_optimal_threshold(y_true, y_proba) -> pd.DataFrame:
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = [compute_cost(y_true, y_proba, t) for t in thresholds]
    df_results = pd.DataFrame(results)
    return df_results.sort_values("total_cost")


def main():
    pipeline = joblib.load("models/churn_pipeline.pkl")

    df = load_data()
    df = clean_data(df)
    df = add_business_features(df)
    _, X_test, _, y_test = split_data(df)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results = find_optimal_threshold(y_test.values, y_proba)
    print("Thresholds sorted by increasing total cost (best first):\n")
    print(results.to_string(index=False))

    best = results.iloc[0]
    default = results[results["threshold"] == 0.50].iloc[0]

    print(f"\nDefault threshold (0.5)  : total cost = {default['total_cost']:.0f}$")
    print(f"Optimal threshold ({best['threshold']}) : total cost = {best['total_cost']:.0f}$")
    savings = default["total_cost"] - best["total_cost"]
    print(f"Potential savings        : {savings:.0f}$ on this test set "
          f"({len(y_test)} customers)")

    results.to_csv("models/threshold_analysis.csv", index=False)
    print("\nAnalysis saved: models/threshold_analysis.csv")


if __name__ == "__main__":
    main()