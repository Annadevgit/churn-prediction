import pandas as pd

def adapt_telco(input_path="data/raw/telco_raw.csv", output_path="data/raw/churn_data.csv"):
    df = pd.read_csv(input_path)

    df = df.rename(columns={
        "customerID": "customer_id",
        "tenure": "tenure_months",
        "Contract": "contract_type",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "InternetService": "internet_service",
        "TechSupport": "tech_support",
        "OnlineSecurity": "online_security",
        "PaymentMethod": "payment_method",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "PaperlessBilling": "paperless_billing",
        "Churn": "churn",
    })

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})


    keep_cols = [
        "customer_id", "tenure_months", "contract_type", "monthly_charges",
        "total_charges", "internet_service", "tech_support", "online_security",
        "payment_method", "senior_citizen", "partner", "dependents"
        , "paperless_billing", "churn",
    ]
    df = df[keep_cols]
    df.to_csv(output_path, index=False)
    print(f"Adapté : {output_path} — shape {df.shape}")

if __name__ == "__main__":
    adapt_telco()
