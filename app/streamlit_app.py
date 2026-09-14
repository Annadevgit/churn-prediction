"""Streamlit interface to test the churn prediction model live."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from features import add_business_features

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write(
    "Enter a customer's profile to estimate their probability of churning "
    "and understand the risk factors."
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline.pkl"

if not MODEL_PATH.exists():
    st.error(
        "Model not found. Run first:\n\n"
        "```\npython -m src.generate_synthetic_data\npython -m src.train\n```"
    )
    st.stop()

pipeline = joblib.load(MODEL_PATH)

col1, col2 = st.columns(2)

with col1:
    tenure_months = st.slider("Tenure (months)", 0, 72, 12)
    contract_type = st.selectbox(
        "Contract type", ["Month-to-month", "One year", "Two year"]
    )
    monthly_charges = st.number_input("Monthly charges ($)", 18.0, 150.0, 65.0)
    total_charges = st.number_input(
        "Total charges ($)", 0.0, 10000.0, float(monthly_charges * tenure_months)
    )
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    

with col2:
    tech_support = st.selectbox("Tech support subscribed", ["Yes", "No"])
    online_security = st.selectbox("Online security subscribed", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior (65+)", [0, 1])
    partner = st.selectbox("Has a partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

if st.button("Predict churn risk", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "tenure_months": tenure_months,
                "contract_type": contract_type,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "internet_service": internet_service,
                "tech_support": tech_support,
                "online_security": online_security,
                "payment_method": payment_method,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
                "paperless_billing": paperless_billing,
            }
        ]
    )
    input_df = add_business_features(input_df)

    proba = pipeline.predict_proba(input_df)[0, 1]

    st.divider()
    st.metric("Churn probability", f"{proba:.1%}")

    if proba >= 0.7:
        st.error("🔴 High risk — retention action recommended")
    elif proba >= 0.4:
        st.warning("🟠 Moderate risk — worth monitoring")
    else:
        st.success("🟢 Low risk")

st.divider()
st.caption(
    "Demo project. "
    "See the repo README for the full methodology."
)