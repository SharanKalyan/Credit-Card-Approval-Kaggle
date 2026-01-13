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
# Animated Background Function
# --------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        @keyframes fadeZoom {{
            0% {{
                opacity: 0;
                transform: scale(1.05);
            }}
            100% {{
                opacity: 1;
                transform: scale(1);
            }}
        }}

        .stApp {{
            background-image:
            linear-gradient(
                rgba(255,255,255,0),
                rgba(255,255,255,0)
            ),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            animation: fadeZoom 0.9s ease-in-out;
        }}

        html, body, [class*="css"] {{
            color: #000000 !important;
        }}

        div[data-testid="stFormSubmitButton"] button {{
            background-color: #1f4fd8 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: none !important;
        }}

        div[data-testid="stFormSubmitButton"] button:hover {{
            background-color: #163bb5 !important;
        }}

        div[data-baseweb="select"] > div {{
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 6px !important;
        }}

        ul[role="listbox"] li {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Initial Landing Background
# --------------------------------------------------
add_bg_from_local("landingpage.png")

# --------------------------------------------------
# Title & Hero Section
# --------------------------------------------------
st.title("💳 Credit Card Approval Prediction App")

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h2 style="color:#1f4fd8;">
            Check your credit card eligibility here
        </h2>
        <p style="font-size:16px; color:#fcf2f2;">
            This model estimates <b>credit card approval probability</b>
            using historical applicant patterns.<br>
            Designed to simulate <b>early-stage credit risk screening</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info("🔒 This is a demo ML application. No data is stored.")
st.markdown("---")

# --------------------------------------------------
# Model Information (Expandable Section)
# --------------------------------------------------
with st.expander("ℹ️ Model Information & Decision Logic"):
    st.markdown(
        """
        ### 🧠 Model Overview
        - **Model Used:** Random Forest Classifier  
        - Trained on historical credit card application data  
        - Designed for **early-stage credit approval screening**  

        ### 📊 Model Performance (Validation Set)
        - **Recall:** 0.98  
        - **Precision:** 0.99  
        - **F1 Score:** 0.99  

        **How to interpret this:**
        - High **recall** ensures most eligible applicants are correctly identified  
        - Strong **precision** reduces false approvals  
        - Balanced **F1 score** indicates stable real-world performance  

        ### 🧾 Decision Interpretation (Based on Approval Probability)
        - **0 – 50% → ❌ Reject**  
          High risk profile, unlikely to meet approval criteria  

        - **50 – 80% → ⚠️ Flag for Manual Review**  
          Borderline cases requiring human judgment or additional checks  

        - **80%+ → ✅ High Confidence Approval**  
          Strong applicant profile with high confidence  

        > ⚠️ *This tool is for demonstration purposes only and does not replace official credit risk assessments.*
        """
    )

st.markdown("---")



# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model(path="random_forest_quantised.pkl"):
    return joblib.load(path)

model_path = Path("random_forest_quantised.pkl")

if not model_path.exists():
    st.error("❌ Model file not found.")
    st.stop()

model = load_model(str(model_path))

# --------------------------------------------------
# Features
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
# Individual Prediction
# --------------------------------------------------
st.header("🧍 Individual Credit Check")

with st.form("credit_form"):
    gender = st.selectbox("Applicant Gender", ["Male", "Female"])
    age = st.number_input("Applicant Age", min_value=18, value=25, step=1)
    income = st.number_input("Total Income", min_value=0, value=100000, step=10000)
    good_debt = st.number_input("Total Number of Good Debts", min_value=0, value=10)
    bad_debt = st.number_input("Total Number of Bad Debts", min_value=0, value=5)

    submitted = st.form_submit_button("Predict")

if submitted:
    gender_encoded = 1 if gender == "Male" else 0
    debt_score = good_debt - bad_debt

    X = pd.DataFrame(
        [[gender_encoded, age, income, good_debt, bad_debt, debt_score]],
        columns=FEATURES
    )

    prob = model.predict_proba(X)[0][1]
    prediction = model.predict(X)[0]  # 1 = Approved, 0 = Rejected

    # 🔁 Animated background switch
    if prediction == 1:
        add_bg_from_local("landingpage.png")
    else:
        add_bg_from_local("landingpage.png")

    if prob >= 0.75:
        st.success(f"✅ **HIGH CONFIDENCE APPROVAL**\n\nApproval Probability: **{prob:.2f}**")
    elif prob >= 0.5:
        st.warning(f"⚠️ **BORDERLINE – MANUAL REVIEW ADVISED**\n\nApproval Probability: **{prob:.2f}**")
    else:
        st.error(f"❌ **HIGH RISK – REJECTION LIKELY**\n\nApproval Probability: **{prob:.2f}**")

# --------------------------------------------------
# Batch Prediction (UNCHANGED)
# --------------------------------------------------
st.markdown("---")
st.header("📂 Batch Credit Check (CSV Upload)")

st.info(
    "Upload a CSV to evaluate multiple applicants at once. "
    "This simulates a real-world bank pre-screening workflow."
)

sample_df = pd.DataFrame({
    "Applicant_Gender": ["Male", "Female"],
    "Applicant_Age": [30, 45],
    "Total_Income": [500000, 800000],
    "Total_Good_Debt": [5, 8],
    "Total_Bad_Debt": [1, 2]
})

st.download_button(
    "⬇️ Download Sample CSV",
    sample_df.to_csv(index=False),
    "sample_credit_applicants.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = sample_df.columns.tolist()
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    df["Applicant_Gender"] = df["Applicant_Gender"].map({"Male": 1, "Female": 0})
    df["Debt_Score"] = df["Total_Good_Debt"] - df["Total_Bad_Debt"]

    X_batch = df[FEATURES]
    df["Approval_Probability"] = model.predict_proba(X_batch)[:, 1]
    df["Decision"] = df["Approval_Probability"].apply(
        lambda x: "HIGH APPROVAL" if x >= 0.75 else
                  "BORDERLINE" if x >= 0.5 else
                  "HIGH RISK"
    )

    st.dataframe(df)

    st.download_button(
        "⬇️ Download Predictions",
        df.to_csv(index=False),
        "credit_predictions.csv",
        "text/csv"
    )




