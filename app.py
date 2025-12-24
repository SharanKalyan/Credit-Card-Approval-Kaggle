import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sklearn
st.write("scikit-learn version:", sklearn.__version__)

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("Credit Risk Prediction App")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model(path="random_forest_quantised.pkl"):
    return joblib.load(path)

model_path = Path("random_forest_quantised.pkl")

if not model_path.exists():
    st.error("❌ Model file not found. Upload random_forest_quantised.pkl")
    st.stop()

model = load_model(str(model_path))

# --------------------------------------------------
# Features used by model
# --------------------------------------------------
FEATURES = [
    "Applicant_Gender",
    "Applicant_Age",
    "Total_Income",
    "Total_Good_Debt",
    "Total_Bad_Debt",
    "Debt_Score"
]

# --------------------------------------------------
# UI Inputs (ONLY THESE 5)
# --------------------------------------------------
with st.form("credit_form"):

    gender = st.radio("Applicant Gender", ["Male", "Female"])
    age = st.number_input("Applicant Age", min_value=18, step=1)
    income = st.number_input("Total Income", min_value=0.0)
    good_debt = st.number_input("Total Good Debt", min_value=0.0)
    bad_debt = st.number_input("Total Bad Debt", min_value=0.0)

    submitted = st.form_submit_button("Predict")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if submitted:
    try:
        # Encode Gender
        gender_encoded = 1 if gender == "Male" else 0

        # Create Debt Score
        debt_score = good_debt - bad_debt

        # Build input DataFrame
        X = pd.DataFrame(
            [[
                gender_encoded,
                age,
                income,
                good_debt,
                bad_debt,
                debt_score
            ]],
            columns=FEATURES
        )

        # Predict
        prob = model.predict_proba(X)[0][1]

        if prob >= 0.7:
            st.success(
                f"✅ APPROVED\n\n"
                f"Approval Probability: {prob:.2f}\n\n"
                f"Debt Score: {debt_score}"
            )
        else:
            st.error(
                f"❌ REJECTED\n\n"
                f"Risk Probability: {1 - prob:.2f}\n\n"
                f"Debt Score: {debt_score}"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")




