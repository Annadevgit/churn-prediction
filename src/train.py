"""Train and compare multiple churn prediction models."""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_processing import clean_data, load_data, split_data
from src.features import add_business_features, build_preprocessor

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision"}


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def evaluate_models(X_train, y_train) -> dict:
    """Compare models using stratified cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        start = time.time()
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1
        )
        elapsed = time.time() - start

        results[name] = {
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "avg_precision_mean": float(np.mean(scores["test_average_precision"])),
            "training_time_sec": round(elapsed, 2),
        }
        print(
            f"{name:20s} | ROC-AUC: {results[name]['roc_auc_mean']:.4f} "
            f"(+/- {results[name]['roc_auc_std']:.4f}) | "
            f"PR-AUC: {results[name]['avg_precision_mean']:.4f} | "
            f"{elapsed:.1f}s"
        )

    return results


def train_final_model(X_train, y_train, X_test, y_test, best_model_name: str):
    """Train the selected model on the entire training set and evaluate it on the test set."""
    pipeline = build_pipeline(MODELS[best_model_name])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return pipeline, test_metrics


def main():
    print("Data Loading and Cleaning...")
    df = load_data()
    df = clean_data(df)
    df = add_business_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    print("Model Comparison (5-fold cross-validation)...")
    cv_results = evaluate_models(X_train, y_train)

    best_model_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    print(f"\nBest Model : {best_model_name}")

    print("\nFinal training and evaluation on the test set...")
    pipeline, test_metrics = train_final_model(
        X_train, y_train, X_test, y_test, best_model_name
    )
    print(f"ROC-AUC (test) : {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC (test)  : {test_metrics['average_precision']:.4f}")

    # Sauvegarde
    joblib.dump(pipeline, "models/churn_pipeline.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump(
            {"cv_results": cv_results, "best_model": best_model_name, "test_metrics": test_metrics},
            f,
            indent=2,
        )
    print("\nModel saved in models/churn_pipeline.pkl")
    print("Metrics saved in models/metrics.json")


if __name__ == "__main__":
    main()
