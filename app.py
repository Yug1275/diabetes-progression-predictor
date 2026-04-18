# =============================================================
# app.py — Streamlit Web App for Diabetes Risk Prediction
# =============================================================
# Run with: streamlit run app.py
# =============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Custom CSS Styling — Fixed text colors
# -------------------------------------------------------
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }

    .stButton > button {
        background-color: #e74c3c;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #c0392b;
        color: white;
    }

    /* FIX: Explicit dark text colors for result cards */
    .result-high {
        background-color: #fde8e8;
        border-left: 6px solid #e74c3c;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2c3e50 !important;
    }
    .result-high h2 {
        color: #c0392b !important;
    }
    .result-high p {
        color: #2c3e50 !important;
    }

    .result-low {
        background-color: #e8fdf0;
        border-left: 6px solid #2ecc71;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2c3e50 !important;
    }
    .result-low h2 {
        color: #1a7a4a !important;
    }
    .result-low p {
        color: #2c3e50 !important;
    }
    
    .result-moderate {
        background-color: #fff4e6;
        border-left: 6px solid #f39c12;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2c3e50 !important;
    }
    .result-moderate h2 {
        color: #e67e22 !important;
    }
    .result-moderate p {
        color: #2c3e50 !important;
    }

    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Load Model, Scaler, Feature Names
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base          = os.path.dirname(os.path.abspath(__file__))
    model         = joblib.load(os.path.join(base, 'models', 'final_best_model.pkl'))
    scaler        = joblib.load(os.path.join(base, 'models', 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(base, 'models', 'feature_names.pkl'))
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def get_engineered_features(pregnancies, glucose, bp,
                             skin, insulin, bmi, dpf, age):
    """Recreate the 4 engineered features from Phase 4."""
    bmi_cat     = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else 3))
    age_grp     = 0 if age < 30   else (1 if age < 45 else 2)
    gluc_lvl    = 0 if glucose < 100 else (1 if glucose < 126 else 2)
    insulin_res = bmi * glucose / 1000
    return bmi_cat, age_grp, gluc_lvl, insulin_res


def make_prediction(inputs):
    """
    Scale inputs using a named DataFrame to avoid
    'feature names' warning, then return prediction + probabilities.
    """
    eng = get_engineered_features(*inputs)

    # Combine base + engineered features into a list
    all_values = list(inputs) + list(eng)

    # ✅ FIX: wrap in DataFrame with correct feature names
    #    so StandardScaler doesn't raise the UserWarning
    input_df = pd.DataFrame([all_values], columns=feature_names)

    scaled      = scaler.transform(input_df)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]
    return prediction, probability


def plot_risk_gauge(risk_pct):
    """Draw a semicircular gauge chart."""
    fig, ax = plt.subplots(figsize=(5, 3),
                           subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    # Background arc
    arc_bg = mpatches.Wedge(
        center=(0.5, 0), r=0.45,
        theta1=0, theta2=180,
        width=0.15, color='#ecf0f1'
    )
    ax.add_patch(arc_bg)

    # Risk arc
    risk_theta = risk_pct / 100 * 180
    color = ('#e74c3c' if risk_pct > 60 else
             '#f39c12' if risk_pct > 40 else '#2ecc71')
    arc_risk = mpatches.Wedge(
        center=(0.5, 0), r=0.45,
        theta1=0, theta2=risk_theta,
        width=0.15, color=color, alpha=0.85
    )
    ax.add_patch(arc_risk)

    # Text
    ax.text(0.5, 0.15, f'{risk_pct:.1f}%',
            ha='center', va='center',
            fontsize=22, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.02, 'Diabetes Risk',
            ha='center', va='center',
            fontsize=11, color='gray',
            transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.6)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
    st.title("🩺 About This App")
    st.markdown("""
    This tool uses **Machine Learning** to predict the
    risk of early diabetes progression based on
    medical diagnostic data.

    **How to use:**
    1. Enter patient medical values on the right
    2. Click **Predict Risk** button
    3. View the risk assessment result

    ---
    **Dataset:** PIMA Indians Diabetes Dataset
    **Model:** Trained on 768 patient records

    ---
    ⚠️ **Disclaimer**
    This tool is for educational purposes only
    and does NOT replace professional medical advice.
    """)

    st.markdown("---")
    st.markdown("**📊 Model Performance**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy",  "~85%")
    c2.metric("ROC-AUC",   "~90%")
    c1.metric("Recall",    "~83%")
    c2.metric("F1-Score",  "~84%")

    st.markdown("---")
    st.markdown("**🎓 AI Lab Project — 6th Sem**")


# -------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------
st.title("🩺 Early Diabetes Progression Predictor")
st.markdown("##### AI-powered diabetes risk assessment using Machine Learning")
st.markdown("---")

# -------------------------------------------------------
# INPUT SECTION
# -------------------------------------------------------
st.markdown('<p class="section-header">📋 Patient Medical Information</p>',
            unsafe_allow_html=True)
st.markdown("Fill in the patient's medical values below:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔢 Basic Information**")
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1,
        help="Number of times pregnant"
    )
    age = st.number_input(
        "Age (years)", min_value=10, max_value=100, value=30, step=1,
        help="Patient age in years"
    )
    bmi = st.number_input(
        "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index"
    )

with col2:
    st.markdown("**🩸 Blood Measurements**")
    glucose = st.number_input(
        "Glucose (mg/dL)", min_value=50, max_value=250, value=100, step=1,
        help="Plasma glucose concentration after 2hr oral glucose test"
    )
    insulin = st.number_input(
        "Insulin (mu U/ml)", min_value=10, max_value=600, value=80, step=1,
        help="2-Hour serum insulin level"
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)", min_value=30, max_value=140, value=70, step=1,
        help="Diastolic blood pressure"
    )

with col3:
    st.markdown("**📏 Other Measurements**")
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", min_value=5, max_value=80, value=20, step=1,
        help="Triceps skin fold thickness"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.05, max_value=2.5,
        value=0.5, step=0.01,
        help="Genetic likelihood score based on family history"
    )

    # BMI info box
    bmi_label = ("Underweight" if bmi < 18.5 else
                 "Normal"      if bmi < 25   else
                 "Overweight"  if bmi < 30   else "Obese")
    bmi_color = ("#3498db" if bmi < 18.5 else
                 "#2ecc71" if bmi < 25   else
                 "#f39c12" if bmi < 30   else "#e74c3c")
    st.markdown(f"""
        <div style='background:{bmi_color}22; border-left:4px solid {bmi_color};
                    padding:10px; border-radius:8px; margin-top:28px;'>
            <b style='color:#2c3e50;'>BMI Category:</b><br>
            <span style='color:{bmi_color}; font-size:18px; font-weight:bold;'>
                {bmi_label}
            </span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# SAMPLE PATIENT BUTTONS
# -------------------------------------------------------
st.markdown("**💡 Try a Sample Patient:**")
samp_col1, samp_col2, samp_col3 = st.columns(3)

if 'sample' not in st.session_state:
    st.session_state.sample = None

if samp_col1.button("🔴 High Risk Patient"):
    st.session_state.sample = 'high'
    st.rerun()
if samp_col2.button("🟢 Low Risk Patient"):
    st.session_state.sample = 'low'
    st.rerun()
if samp_col3.button("🔄 Reset Values"):
    st.session_state.sample = None
    st.rerun()

if st.session_state.sample == 'high':
    st.info("💡 Set values → Pregnancies:6, Glucose:148, BP:72, Skin:35, Insulin:125, BMI:33.6, DPF:0.627, Age:50 — then click Predict")
elif st.session_state.sample == 'low':
    st.info("💡 Set values → Pregnancies:1, Glucose:89, BP:66, Skin:23, Insulin:94, BMI:28.1, DPF:0.167, Age:21 — then click Predict")

st.markdown("---")

# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------
predict_btn = st.button("🔬 Predict Diabetes Risk", width='stretch')

if predict_btn:
    inputs = (pregnancies, glucose, blood_pressure,
              skin_thickness, insulin, bmi, dpf, age)

    with st.spinner("Analyzing medical data..."):
        prediction, probability = make_prediction(inputs)

    risk_pct    = probability[1] * 100
    no_risk_pct = probability[0] * 100

    st.markdown("---")
    st.markdown('<p class="section-header">🔬 Prediction Result</p>',
                unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if risk_pct > 75:
        # 🔴 HIGH RISK
            st.markdown(f"""
                <div class="result-high">
                    <h2 style="color:#c0392b;">🔴 HIGH RISK OF DIABETES</h2>
                    <p style="font-size:16px; color:#2c3e50;">
                        The model detected <b>high risk indicators</b>
                        for diabetes progression.<br><br>
                        <b>Confidence:</b> {risk_pct:.1f}%<br>
                        <b>Risk Level:</b> 🔴 High
                    </p>
                    <hr style="border-color:#e74c3c33;">
                    <p style="color:#2c3e50;">
                        ⚕️ <b>Recommendation:</b> Please consult a healthcare
                        professional for a comprehensive diabetes screening
                        including HbA1c and fasting glucose tests.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        elif risk_pct > 40:
            # 🟡 MODERATE RISK
            st.markdown(f"""
                <div class="result-moderate">
                    <h2>🟡 MODERATE RISK OF DIABETES</h2>
                    <p style="font-size:16px; color:#2c3e50;">
                        The model detected <b>moderate risk indicators</b>
                        for diabetes progression.<br><br>
                        <b>Confidence:</b> {risk_pct:.1f}%<br>
                        <b>Risk Level:</b> 🟡 Moderate
                    </p>
                    <hr style="border-color:#f39c1233;">
                    <p style="color:#2c3e50;">
                        ⚕️ <b>Recommendation:</b> Monitor your health regularly,
                        improve lifestyle habits, and consider consulting a doctor
                        if risk factors persist.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        else:
            # 🟢 LOW RISK
            st.markdown(f"""
                <div class="result-low">
                    <h2 style="color:#1a7a4a;">🟢 LOW RISK OF DIABETES</h2>
                    <p style="font-size:16px; color:#2c3e50;">
                        The model found <b>low risk indicators</b>
                        for diabetes progression.<br><br>
                        <b>Confidence:</b> {no_risk_pct:.1f}%<br>
                        <b>Risk Level:</b> 🟢 Low
                    </p>
                    <hr style="border-color:#2ecc7133;">
                    <p style="color:#2c3e50;">
                        ⚕️ <b>Recommendation:</b> Maintain a healthy lifestyle
                        with regular exercise, balanced diet, and annual
                        health check-ups.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    
        st.markdown("**📊 Probability Breakdown:**")
        st.progress(int(no_risk_pct),
                    text=f"Non-Diabetic: {no_risk_pct:.1f}%")
        st.progress(int(risk_pct),
                    text=f"Diabetic: {risk_pct:.1f}%")

    with res_col2:
        # Gauge chart — ✅ FIX: replaced use_container_width with width='content'
        gauge_fig = plot_risk_gauge(risk_pct)
        st.pyplot(gauge_fig, width='content')

        # Input summary table — ✅ FIX: width='stretch'
        st.markdown("**📋 Input Summary:**")
        summary_data = {
            'Feature'     : ['Glucose', 'BMI', 'Age',
                             'Blood Pressure', 'Insulin'],
            'Your Value'  : [glucose, bmi, age,
                             blood_pressure, insulin],
            'Normal Range': ['70–99', '18.5–24.9', '—',
                             '60–80', '16–166']
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            width='stretch',
            hide_index=True
        )

    st.markdown("---")
    st.warning(
        "⚠️ **Medical Disclaimer:** This prediction is generated by an AI model "
        "trained on historical data. It is intended for **educational purposes only** "
        "and does **NOT** constitute medical advice. Always consult a qualified "
        "healthcare professional for medical decisions."
    )