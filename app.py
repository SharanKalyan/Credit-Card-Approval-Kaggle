import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.title("Credit Risk Prediction App")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(path="random_forest.pkl"):
    return joblib.load(path)

model_path = Path("random_forest.pkl")

if not model_path.exists():
    st.error("Model file not found. Upload random_forest.pkl to the repo.")
    st.stop()
else:
    model = load_model(str(model_path))


# -------------------------------
# Feature List
# -------------------------------
FEATURES = [
    'Applicant_Gender', 'Owned_Car', 'Owned_Realty', 'Total_Children',
    'Total_Income', 'Total_Income_Catg', 'Income_Type', 'Education_Type',
    'Family_Status', 'Housing_Type', 'Owned_Mobile_Phone',
    'Owned_Work_Phone', 'Owned_Phone', 'Owned_Email', 'Job_Title',
    'Total_Family_Members', 'Applicant_Age', 'Years_of_Working',
    'Total_Bad_Debt', 'Total_Bad_Debt_Catg', 'Total_Good_Debt',
    'Debt_Score', 'Debit_Score_Catg'
]

st.header("Make a Prediction")

# -------------------------------
# Build inputs dynamically
# -------------------------------
input_data = {}

with st.form("predict_form"):

    # CATEGORICAL FEATURES (dropdowns)
    categorical_cols = [
        'Applicant_Gender', 'Owned_Car', 'Owned_Realty', 
        'Total_Income_Catg', 'Income_Type', 'Education_Type',
        'Family_Status', 'Housing_Type', 'Owned_Mobile_Phone',
        'Owned_Work_Phone', 'Owned_Phone', 'Owned_Email', 
        'Job_Title', 'Total_Bad_Debt_Catg', 'Debit_Score_Catg'
    ]

    # NUMERIC FEATURES (number_input)
    numeric_cols = [
        'Total_Children', 'Total_Income', 'Total_Family_Members',
        'Applicant_Age', 'Years_of_Working', 'Total_Bad_Debt',
        'Total_Good_Debt', 'Debt_Score'
    ]

    st.subheader("Categorical Inputs")

    for col in categorical_cols:
        input_data[col] = st.text_input(col, value="")

    st.subheader("Numeric Inputs")

    for col in numeric_cols:
        input_data[col] = st.number_input(col, value=0.0)

    submitted = st.form_submit_button("Predict")


# -------------------------------
# Run Prediction
# -------------------------------
if submitted:
    try:
        row = [input_data[col] for col in FEATURES]
        X = pd.DataFrame([row], columns=FEATURES)

        pred = model.predict(X)

        st.success(f"Prediction: {pred[0]}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write("Probabilities:", proba.tolist())

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# -------------------------------
# Batch Prediction via CSV
# -------------------------------
st.write("---")
st.header("Batch Prediction from CSV")

uploaded = st.file_uploader("Upload CSV with same feature columns", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    try:
        preds = model.predict(df[FEATURES])
        df["prediction"] = preds
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions (CSV)", csv, "preds.csv")

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
