"""Model Explainability with SHAP: Global and Local Importance of Features."""

import joblib
import matplotlib.pyplot as plt
import shap

from src.data_processing import clean_data, load_data, split_data
from src.features import add_business_features


def main():
    print("data and model loading...")
    pipeline = joblib.load("models/churn_pipeline.pkl")

    df = load_data()
    df = clean_data(df)
    df = add_business_features(df)
    _, X_test, _, _ = split_data(df)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    print("Calculation of SHAP Values...")
    explainer = shap.Explainer(model, X_test_transformed, feature_names=feature_names)
    shap_values = explainer(X_test_transformed[:500])  

    # globale Importance
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed[:500], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150)
    print("Graph saved : models/shap_summary.png")

    # Top features by average absolute value
    import numpy as np

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    print("\nTop 10 Most Influential Features :")
    for i in top_idx:
        print(f"  {feature_names[i]:40s} {mean_abs_shap[i]:.4f}")


if __name__ == "__main__":
    main()
