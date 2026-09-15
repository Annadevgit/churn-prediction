# 📉 Customer Churn Prediction

**Live demo**: https://churn-prediction-gqf6tmk8cgsbffxlqtzpsr.streamlit.app/
**Live API docs**: https://churn-prediction-c4bt.onrender.com/docs

An end-to-end machine learning pipeline — from raw data to deployment — that predicts
the probability of customer churn, with model explainability and an estimate of
business impact.

![CI](https://github.com/Annadevgit/churn-prediction/actions/workflows/ci.yml/badge.svg)

## Context and problem statement

In subscription-based industries (telecom, SaaS, banking, insurance), acquiring a new
customer typically costs 5 to 7 times more than retaining an existing one. This project
addresses a simple operational question: **which customers are most likely to churn,
and why?**

The goal isn't just a good classification score — it's a tool a retention team could
actually use: a risk score, an explanation of the driving factors, and an interface to
test scenarios.

## Data

This project uses the **[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**
dataset (IBM sample dataset, hosted on Kaggle), containing ~7,000 telecom customers
with their subscription details and whether they churned.

**Raw column mapping applied** (see `src/rename_columns.py`):

| Original Telco column | Pipeline column |
|---|---|
| `customerID` | `customer_id` |
| `tenure` | `tenure_months` |
| `Contract` | `contract_type` |
| `MonthlyCharges` | `monthly_charges` |
| `TotalCharges` | `total_charges` |
| `InternetService` | `internet_service` |
| `TechSupport` | `tech_support` |
| `OnlineSecurity` | `online_security` |
| `PaymentMethod` | `payment_method` |
| `SeniorCitizen` | `senior_citizen` |
| `Partner` | `partner` |
| `Dependents` | `dependents` |
| `PaperlessBilling` | `paperless_billing` |
| `Churn` (Yes/No) | `churn` (1/0) |


| Variable | Description |
|---|---|
| `tenure_months` | Customer tenure in months |
| `contract_type` | Contract type (month-to-month, 1 year, 2 years) |
| `monthly_charges` / `total_charges` | Billing amounts |
| `internet_service`, `tech_support`, `online_security` | Subscribed services |
| `churn` | Target: 1 if the customer churned |

## Methodology

1. **Cleaning** (`src/data_processing.py`): duplicate removal, missing-value handling,
   robust type conversion (the real dataset stores `TotalCharges` as text with blank
   strings for customers with zero tenure — converted to `NaN` then imputed).
2. **Feature engineering** (`src/features.py`): derived business variables (real
   average monthly spend, spend deviation, tenure segments, structural risk profile)
   that add signal beyond the raw columns.
3. **Modeling** (`src/train.py`): comparison of 3 models via stratified 5-fold cross
   validation — logistic regression (interpretable baseline), random forest, gradient
   boosting — with class imbalance handling.
4. **Selection**: the model with the best mean ROC-AUC in CV is retrained on the full
   training set and evaluated on a held-out test set (never seen during CV).
5. **Explainability** (`src/explain.py`): SHAP values for global feature importance and
   individual prediction interpretation.
6. **Deployment**: FastAPI API (`src/api.py`) + Streamlit interface
   (`app/streamlit_app.py`) to test the model under realistic conditions.
7. **Decision threshold optimization** (`src/threshold_optimization.py`): the default
   0.5 threshold has no business justification. This script simulates the real cost of
   a false positive (unnecessarily contacting a customer) vs a false negative (losing a
   customer without intervention) to find the threshold that minimizes total cost.
8. **Probability calibration** (`src/calibration.py`): a good ROC-AUC doesn't guarantee
   that a "70% predicted risk" corresponds to 70% actual observed churn. This script
   compares the raw model to a calibrated version (Platt scaling) via the Brier score.
9. **Gain / lift chart** (`src/lift_chart.py`): translates model performance into
   business language — "targeting the top 20% highest-risk customers captures X% of
   actual churners" — useful for a non-technical audience.


## Results

| Model | ROC-AUC (CV) | PR-AUC (CV) |
|---|---|---|
| `Logistic Regression` | 0.8459 | 0.6538 |
| `Random Forest`       | 0.8454 | 0.6583 |
| `Gradient Boosting`   | 0.8451 | 0.6530 |

**Most influential risk factors** (via SHAP): from `python -m src.explain` :

Top 10 Most Influential Features :

|`num__tenure_months` | 0.5961 |
|---|---|
|`cat__contract_type_Two year` | 0.3189 |
|`cat__contract_type_Month-to-month` | 0.2752 |
|`num__monthly_charges` | 0.2734 |
|`cat__paperless_billing_Yes` | 0.1879 |
|`cat__internet_service_DSL` | 0.1453 |
|`cat__tenure_bucket_0-6m` | 0.1295 |
|`cat__online_security_Yes` | 0.1176 |
|`num__total_charges` | 0.0991 |
|`cat__tenure_bucket_2-4y` | 0.0986 |

**Estimated business impact** (via `src/lift_chart.py`): targeting the top 20% of
customers by risk score captures a majority of actual churners on the test set — see
`models/gain_lift_table.csv` for exact figures.

**Threshold optimization** (via `src/threshold_optimization.py`): given the cost
asymmetry between losing a customer and contacting one unnecessarily, the optimal
threshold is typically well below 0.5 — the model should err on the side of flagging
too many at-risk customers rather than too few.

## Limitations and possible improvements

- **No temporal dimension**: a production churn system should capture behavioral
  trends over time, not just a single snapshot.
- **Simplified cost assumptions**: the costs used in `threshold_optimization.py`
  (contact cost, customer lifetime value) are estimates — replace with real
  finance/CRM figures for production use.
- **Next step under consideration**: a survival model (Cox regression, or
  time-to-event) to predict *when* a customer is likely to leave, not just *whether*.

## Installation and usage

```bash
git clone <repo-url>
cd churn-prediction
pip install -r requirements.txt

# 1. Place the real data Telco CSV in data/raw/telco_raw.csv, then rename the columns
python -m src.rename_columns

# 2. Train and compare models
python -m src.train

# 3. Generate SHAP explainability plots
python -m src.explain

# 4. Advanced analyses
python -m src.threshold_optimization   # optimal threshold by business cost
python -m src.calibration              # quality of predicted probabilities
python -m src.lift_chart               # gain curve / business impact

# 5. Run the API
uvicorn src.api:app --reload
# -> interactive docs at http://localhost:8000/docs

# 6. Run the Streamlit interface
streamlit run app/streamlit_app.py
```

**With Docker:**
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

**Tests:**
```bash
pytest tests/ -v
```

## Project structure

```
churn-prediction/
├── data/
│   ├── raw/                     # Raw data (not versioned, except .gitkeep)
│   └── processed/
├── notebooks/                   # Ad hoc exploration only
├── src/
│   ├── rename_columns.py        # Improve variable names readability
│   ├── data_processing.py       # Cleaning
│   ├── features.py              # Feature engineering + preprocessor
│   ├── train.py                 # Model training and comparison
│   ├── explain.py                # SHAP explainability
│   ├── threshold_optimization.py # Optimal threshold by business cost
│   ├── calibration.py           # Probability calibration
│   ├── lift_chart.py            # Gain curve / business impact
│   └── api.py                   # FastAPI API
├── app/
│   └── streamlit_app.py         # Demo interface
├── tests/
│   └── test_pipeline.py
├── models/                      # Saved models and metrics
├── .github/workflows/ci.yml     # Automated tests on every push
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech stack

Python · pandas · scikit-learn · SHAP · FastAPI · Streamlit · Docker · pytest · GitHub Actions

