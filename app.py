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
# Background Image + THEME FIXES
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

        div[data-baseweb="select"] span {{
            color: #000000 !important;
        }}

        div[data-baseweb="select"] svg {{
            fill: #000000 !important;
        }}

        ul[role="listbox"] li {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}

        ul[role="listbox"] li:hover {{
            background-color: #f2f4f8 !important;
        }}

        div[data-testid="stFileUploader"] button {{
            background-color: #1f4fd8 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("Banking.png")

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
        <p style="font-size:16px; color:#444;">
            This model estimates <b>credit card approval probability</b> 
            using historical applicant patterns.<br>
            Designed to simulate <b>early-stage credit risk screening</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    🔗 **Full implementation & methodology:**  
    [Credit Card Approval System – GitHub](https)
