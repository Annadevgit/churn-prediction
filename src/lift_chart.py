"""
Gain chart and lift chart.

Answers the concrete business question: "if we target the top X% of customers
by score, what proportion of actual churners do we capture?"
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_processing import clean_data, load_data, split_data
from src.features import add_business_features


def compute_gain_lift(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)

    total_positives = df["y_true"].sum()
    n = len(df)

    deciles = []
    for decile in range(1, 11):
        cutoff = int(n * decile / 10)
        subset = df.iloc[:cutoff]
        captured_positives = subset["y_true"].sum()
        gain = captured_positives / total_positives
        random_gain = decile / 10
        lift = gain / random_gain if random_gain > 0 else np.nan

        deciles.append({
            "decile": decile,
            "pct_customers_contacted": decile * 10,
            "pct_churners_captured": round(gain * 100, 1),
            "lift": round(lift, 2),
        })

    return pd.DataFrame(deciles)


def plot_gain_chart(gain_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        gain_df["pct_customers_contacted"], gain_df["pct_churners_captured"],
        marker="o", label="Model", color="#2563eb",
    )
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random targeting")

    ax.set_xlabel("% of customers contacted (sorted by decreasing risk score)")
    ax.set_ylabel("% of actual churners captured")
    ax.set_title("Gain chart")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")


def main():
    pipeline = joblib.load("models/churn_pipeline.pkl")

    df = load_data()
    df = clean_data(df)
    df = add_business_features(df)
    _, X_test, _, y_test = split_data(df)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    gain_df = compute_gain_lift(y_test.values, y_proba)
    print(gain_df.to_string(index=False))

    top20 = gain_df[gain_df["pct_customers_contacted"] == 20].iloc[0]
    print(
        f"\nBy contacting the top 20% highest-risk customers, we capture "
        f"{top20['pct_churners_captured']}% of actual churners "
        f"(a {top20['lift']}x lift vs random targeting)."
    )

    plot_gain_chart(gain_df, "models/gain_chart.png")
    gain_df.to_csv("models/gain_lift_table.csv", index=False)


if __name__ == "__main__":
    main()