"""
Calibration of predicted probabilities.

A model can have an excellent ROC-AUC (good ranking power) while still being
poorly calibrated (when it says "70% risk", it's not really 70% of observed
cases). For a business use case (prioritizing actions based on a score),
calibration matters just as much as discrimination.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.data_processing import clean_data, load_data, split_data
from src.features import add_business_features


def plot_calibration(y_true, y_proba_raw, y_proba_calibrated, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))

    for y_proba, label in [(y_proba_raw, "Raw model"), (y_proba_calibrated, "Calibrated model")]:
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")


def main():
    pipeline = joblib.load("models/churn_pipeline.pkl")

    df = load_data()
    df = clean_data(df)
    df = add_business_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Raw model probabilities
    y_proba_raw = pipeline.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, y_proba_raw)

    # Calibrated model (sigmoid = Platt scaling, robust on medium-sized datasets)
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    y_proba_calibrated = calibrated.predict_proba(X_test)[:, 1]
    brier_calibrated = brier_score_loss(y_test, y_proba_calibrated)

    print(f"Brier score - raw model        : {brier_raw:.4f}")
    print(f"Brier score - calibrated model : {brier_calibrated:.4f}")
    print("(lower = better; the Brier score measures the quality of the "
          "probabilities, not just the ranking)")

    plot_calibration(y_test.values, y_proba_raw, y_proba_calibrated, "models/calibration_curve.png")

    if brier_calibrated < brier_raw:
        joblib.dump(calibrated, "models/churn_pipeline_calibrated.pkl")
        print("\nThe calibrated model is better -> saved to "
              "models/churn_pipeline_calibrated.pkl")
    else:
        print("\nThe raw model was already well calibrated, no gain from calibration.")


if __name__ == "__main__":
    main()
