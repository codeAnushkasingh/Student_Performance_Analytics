"""
Student Placement Prediction & AI Career Insights
==================================================
Streamlit front-end that:
  1. Accepts student inputs via a clean form
  2. Runs inference using the saved Logistic Regression model + StandardScaler
  3. Calls Google Gemini to explain the result and generate personalised tips
     (gracefully degrades if GEMINI_API_KEY is not set)

Usage:
    streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Prediction & AI Career Insights",
    page_icon="🎓",
    layout="centered",
)

# ── Load model & scaler (cached so they load only once) ──────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model  = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

FEATURE_COLS = [
    "CGPA", "Internships", "Projects", "Workshops_Certifications",
    "AptitudeTestScore", "SoftSkillsRating",
    "ExtracurricularActivities", "PlacementTraining",
    "SSC_Marks", "HSC_Marks",
]

# ── Gemini helper ─────────────────────────────────────────────────────────────
def call_gemini(prompt: str) -> str:
    """Call Google Gemini API (google-genai SDK). Returns empty string on any failure."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from google import genai
        client   = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as exc:
        return f"_GenAI unavailable: {exc}_"


def build_gemini_prompt(inputs: dict, prediction: str, probability: float) -> str:
    return f"""You are a helpful student career advisor.

A student has the following academic and engagement profile:
- CGPA: {inputs['CGPA']}
- SSC Marks (Class 10): {inputs['SSC_Marks']}%
- HSC Marks (Class 12): {inputs['HSC_Marks']}%
- Aptitude Test Score: {inputs['AptitudeTestScore']} / 100
- Soft Skills Rating: {inputs['SoftSkillsRating']} / 5
- Internships completed: {inputs['Internships']}
- Projects completed: {inputs['Projects']}
- Workshops / Certifications: {inputs['Workshops_Certifications']}
- Extracurricular Activities: {'Yes' if inputs['ExtracurricularActivities'] else 'No'}
- Placement Training attended: {'Yes' if inputs['PlacementTraining'] else 'No'}

A machine learning model predicted this student is: **{prediction}**
Placement probability: {probability:.1%}

Please do two things:
1. In 2–3 sentences, explain in plain language why the model may have made this prediction based on the student's profile.
2. Provide exactly 4 personalised, specific, actionable improvement recommendations to help this student improve their placement chances. Format them as a numbered list.

Be encouraging, honest, and concise. Do not repeat the numbers back. Do not say "causes placement"."""


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 1.8rem; font-weight: 700; color: #1f2328; margin-bottom: 2px; }
    .sub-title   { font-size: 1.0rem; color: #57606a; margin-bottom: 20px; }
    .result-box  { padding: 16px 20px; border-radius: 8px; margin: 16px 0; font-size: 1.05rem; }
    .placed      { background: #d1fae5; border-left: 4px solid #059669; color: #065f46; }
    .not-placed  { background: #fee2e2; border-left: 4px solid #dc2626; color: #7f1d1d; }
    .profile-box { background: #f7f8fa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 18px; }
    .disclaimer  { font-size: 0.78rem; color: #6b7280; margin-top: 24px; border-top: 1px solid #e5e7eb; padding-top: 10px; }
    div[data-testid="stForm"] { border: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 Student Placement Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">& AI Career Insights &nbsp;·&nbsp; IBM SkillsBuild Data Analytics with AI</div>', unsafe_allow_html=True)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Enter Student Profile")

    col1, col2 = st.columns(2)

    with col1:
        ssc   = st.number_input("SSC Marks (Class 10) %",    min_value=0.0,  max_value=100.0, value=70.0, step=0.5)
        hsc   = st.number_input("HSC Marks (Class 12) %",    min_value=0.0,  max_value=100.0, value=75.0, step=0.5)
        cgpa  = st.number_input("CGPA (0 – 10)",             min_value=0.0,  max_value=10.0,  value=7.5,  step=0.1)
        apt   = st.number_input("Aptitude Test Score (0–100)",min_value=0,    max_value=100,   value=75,   step=1)
        soft  = st.number_input("Soft Skills Rating (0 – 5)",min_value=0.0,  max_value=5.0,   value=4.0,  step=0.1)

    with col2:
        intern  = st.selectbox("Internships Completed",         options=[0, 1, 2])
        proj    = st.selectbox("Projects Completed",            options=[0, 1, 2, 3])
        ws      = st.selectbox("Workshops / Certifications",    options=[0, 1, 2, 3])
        extra   = st.selectbox("Extracurricular Activities",    options=["Yes", "No"])
        train   = st.selectbox("Attended Placement Training",   options=["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Placement", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "CGPA":                       cgpa,
        "Internships":                intern,
        "Projects":                   proj,
        "Workshops_Certifications":   ws,
        "AptitudeTestScore":          apt,
        "SoftSkillsRating":           soft,
        "ExtracurricularActivities":  1 if extra == "Yes" else 0,
        "PlacementTraining":          1 if train == "Yes" else 0,
        "SSC_Marks":                  ssc,
        "HSC_Marks":                  hsc,
    }

    # Build feature vector in the correct column order
    # DataFrame preserves feature names throughout — avoids sklearn feature-name warnings
    import pandas as _pd
    feature_vector = _pd.DataFrame([[inputs[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    feature_scaled = _pd.DataFrame(scaler.transform(feature_vector), columns=FEATURE_COLS)

    pred_label = model.predict(feature_scaled)[0]
    pred_proba = model.predict_proba(feature_scaled)[0]

    placed      = bool(pred_label == 1)
    probability = float(pred_proba[1])   # probability of "Placed"
    result_text = "Placed ✅" if placed else "Not Placed ❌"
    box_class   = "placed" if placed else "not-placed"

    # ── 1. ML Result ──────────────────────────────────────────────────────
    st.subheader("1. Placement Prediction")
    st.markdown(
        f'<div class="result-box {box_class}">'
        f'<strong>Prediction: {result_text}</strong><br>'
        f'Placement probability: <strong>{probability:.1%}</strong> &nbsp;|&nbsp; '
        f'Not-Placed probability: <strong>{1-probability:.1%}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Probability bar
    st.progress(probability, text=f"Placed probability: {probability:.1%}")

    # ── 2. Student Profile Summary ────────────────────────────────────────
    st.subheader("2. Student Profile Summary")
    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("CGPA",               f"{cgpa:.1f}")
        c2.metric("Aptitude Score",     f"{apt}")
        c3.metric("Soft Skills Rating", f"{soft:.1f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("SSC Marks",          f"{ssc:.0f}%")
        c5.metric("HSC Marks",          f"{hsc:.0f}%")
        c6.metric("Internships",        str(intern))

        c7, c8, c9 = st.columns(3)
        c7.metric("Projects",           str(proj))
        c8.metric("Workshops",          str(ws))
        c9.metric("Extracurricular",    extra)

    # ── 3. GenAI Explanation & Recommendations ────────────────────────────
    st.subheader("3. AI Career Insights")

    api_key_present = bool(os.environ.get("GEMINI_API_KEY", ""))

    if not api_key_present:
        st.info(
            "💡 **GenAI insights are not available.**  \n"
            "Set the `GEMINI_API_KEY` environment variable and restart the app to enable "
            "personalised AI explanations and career recommendations.",
            icon="ℹ️",
        )
    else:
        with st.spinner("Generating AI insights via Gemini…"):
            prompt   = build_gemini_prompt(inputs, result_text.replace(" ✅", "").replace(" ❌", ""), probability)
            ai_reply = call_gemini(prompt)

        if ai_reply:
            st.markdown(ai_reply)
        else:
            st.warning("Could not retrieve AI insights. Please check your GEMINI_API_KEY and network.")

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> ML predictions are estimates '
        'based on the training dataset and are not a guarantee of placement. '
        'Results are for informational purposes only.</div>',
        unsafe_allow_html=True,
    )

# ── Footer (always visible) ───────────────────────────────────────────────────
st.divider()
st.caption(
    "AI-Powered Student Performance & Placement Analytics · "
    "IBM SkillsBuild Data Analytics with AI – BharatCares / AICTE · "
    "Model: Logistic Regression (Test Accuracy 80.85%, ROC-AUC 0.8837)"
)
