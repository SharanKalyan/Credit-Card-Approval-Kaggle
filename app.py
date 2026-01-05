import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Approval Prediction",
    layout="centered"
)

# --------------------------------------------------
# Background Image Styling
# --------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
            linear-gradient(
                rgba(255,255,255,0.4),
                rgba(255,255,255,0.4)
            ),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        button[kind="primary"] {{
            background-color: #1f4fd8;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("banking.png")

# --------------------------------------------------
# Title & Hero Section
# --------------------------------------------------
st.title("💳 Credit Card Approval Prediction App")

st.markdown(
    """
    <div style="text-align:center; padding: 20px 0;">
        <h2 style="color:#1f4fd8;">
            Check your credit card eligibility here
        </h2>
        <p style="font-size:16px; color:#444;">
            Enter a few details to instantly know your approval chances.<br>
            Powered by Machine Learning & Credit Risk Scoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    🔗 **The complete architecture, data pipeline, and model implementation are available in this project’s GitHub repository.**  
    [Credit Card Approval System](https://github.com/SharanKalyan/Credit-Card-Approval-Kaggle)
    """
)

st.info("🔒 This is a demo ML application. No data is stored.")

st.markdown("---")

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
# Features (ORDER MATTERS)
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
# Single Prediction Form
# --------------------------------------------------
st.header("🧍 Individual Credit Check")

with st.form("credit_form"):

    gender = st.selectbox("Applicant Gender", ["Male", "Female"])

    age = st.number_input(
        "Applicant Age",
        min_value=18,
        value=25,
        step=1
    )

    income = st.number_input(
        "Total Income",
        min_value=0,
        value=100000,
        step=10000
    )

    good_debt = st.number_input(
        "Total Number of Good Debts",
        min_value=0,
        value=10,
        step=1
    )

    bad_debt = st.number_input(
        "Total Number of Bad Debts",
        min_value=0,
        value=5,
        step=1
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        gender_encoded = 1 if gender == "Male" else 0
        debt_score = good_debt - bad_debt

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

        prob = model.predict_proba(X)[0][1]

        if prob >= 0.5:
            st.success(
                f"✅ **APPROVED**\n\n"
                f"Approval Probability: **{prob:.2f}**  \n"
                f"Debt Score: **{debt_score}**"
            )
        else:
            st.error(
                f"❌ **REJECTED**\n\n"
                f"Risk Probability: **{1 - prob:.2f}**  \n"
                f"Debt Score: **{debt_score}**"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# Batch Prediction Section
# --------------------------------------------------
st.markdown("---")
st.header("📂 Batch Credit Check (CSV Upload)")

st.info(
    "CSV must contain: Applicant_Gender (Male/Female), Applicant_Age, "
    "Total_Income, Total_Good_Debt, Total_Bad_Debt"
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        required_cols = [
            "Applicant_Gender",
            "Applicant_Age",
            "Total_Income",
            "Total_Good_Debt",
            "Total_Bad_Debt"
        ]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        df["Applicant_Gender"] = df["Applicant_Gender"].map(
            {"Male": 1, "Female": 0}
        )

        if df["Applicant_Gender"].isnull().any():
            st.error("❌ Applicant_Gender must be Male or Female")
            st.stop()

        df["Debt_Score"] = df["Total_Good_Debt"] - df["Total_Bad_Debt"]

        X_batch = df[FEATURES]

        df["Approval_Probability"] = model.predict_proba(X_batch)[:, 1]
        df["Decision"] = df["Approval_Probability"].apply(
            lambda x: "APPROVED" if x >= 0.5 else "REJECTED"
        )

        st.subheader("✅ Batch Prediction Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "credit_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")




