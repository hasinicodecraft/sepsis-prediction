import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    model      = pickle.load(open("model.pkl",     "rb"))
    threshold  = pickle.load(open("threshold.pkl", "rb"))
    model_name = pickle.load(open("model_name.pkl","rb"))
    return model, threshold, model_name

model, threshold, model_name = load_model()

FEATURES = ['age','sex','bmi','systolic_bp','diastolic_bp',
            'glucose','cholesterol','creatinine',
            'diabetes','hypertension','readmission_30d']

st.set_page_config(page_title="SepsisGuard", layout="wide")
st.title("🩺 SepsisGuard — Clinical Sepsis Risk Prediction")
st.caption(f"Model: {model_name}  |  Threshold: {round(threshold*100)}%")
st.markdown("---")

st.subheader("📊 Model Comparison")
comparison = pd.DataFrame([
    {"Technique": "Logistic Regression",         "Accuracy": "74.25%", "Recall": "86.43%", "F1 Score": "64.36%"},
    {"Technique": "Decision Tree",               "Accuracy": "87.70%", "Recall": "83.64%", "F1 Score": "78.53%"},
    {"Technique": "Random Forest",               "Accuracy": "88.65%", "Recall": "84.39%", "F1 Score": "80.00%"},
    {"Technique": "Balanced Random Forest",      "Accuracy": "88.85%", "Recall": "84.94%", "F1 Score": "80.39%"},
    {"Technique": "Gradient Boosting (Best) ✅", "Accuracy": "89.00%", "Recall": "85.13%", "F1 Score": "80.63%"},
])
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.markdown("---")

col_form, col_result = st.columns([3, 2])

with col_form:
    st.subheader("Patient Vitals")
    c1, c2 = st.columns(2)
    with c1:
        age          = st.slider("Age (years)", 18, 90, 55)
        bmi          = st.slider("BMI", 15.0, 45.0, 27.0, step=0.1)
        systolic_bp  = st.number_input("Systolic BP (mmHg)",  70,  180, 120)
        glucose      = st.number_input("Glucose (mg/dL)",     60,  300, 105)
    with c2:
        sex_str      = st.selectbox("Sex", ["Male","Female"])
        sex          = 1 if sex_str == "Male" else 0
        creatinine   = st.number_input("Creatinine (mg/dL)",  0.5, 8.0, 1.0, step=0.1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40,  110,  80)
        cholesterol  = st.number_input("Cholesterol (mg/dL)", 120, 300, 190)

    st.subheader("Medical History")
    h1, h2, h3 = st.columns(3)
    with h1:
        diabetes     = 1 if st.selectbox("Diabetes",          ["No","Yes"]) == "Yes" else 0
    with h2:
        hypertension = 1 if st.selectbox("Hypertension",      ["No","Yes"]) == "Yes" else 0
    with h3:
        readmission  = 1 if st.selectbox("Readmission (30d)", ["No","Yes"]) == "Yes" else 0

    predict = st.button("🔬 Analyse Patient Risk", use_container_width=True, type="primary")

with col_result:
    st.subheader("Prediction Result")
    if predict:
        input_df = pd.DataFrame([[
            age, sex, bmi, systolic_bp, diastolic_bp,
            glucose, cholesterol, creatinine,
            diabetes, hypertension, readmission
        ]], columns=FEATURES)

        prob     = model.predict_proba(input_df)[0][1]
        risk_pct = round(prob * 100, 1)
        high     = prob >= threshold

        if high:
            st.error(f"⚠️ HIGH RISK — {risk_pct}% probability")
            st.warning("Recommend ICU evaluation and sepsis bundle protocol.")
        else:
            st.success(f"✅ LOW RISK — {risk_pct}% probability")
            st.info("Continue standard monitoring.")

        st.markdown("---")
        st.markdown("**Risk factor breakdown:**")
        factors = {
            "Creatinine":  min(1.0, creatinine / 8.0),
            "Glucose":     min(1.0, max(0.0, (glucose - 60) / 240)),
            "Systolic BP": min(1.0, max(0.0, (180 - systolic_bp) / 110)),
            "Age":         min(1.0, (age - 18) / 72),
            "Readmission": float(readmission),
            "Diabetes":    float(diabetes),
        }
        for fname, val in factors.items():
            st.markdown(f"<small style='color:gray'>{fname} — {round(val*100)}%</small>",
                        unsafe_allow_html=True)
            st.progress(val)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  "89.00%")
        m2.metric("Recall",    "85.13%")
        m3.metric("F1 Score",  "80.63%")
    else:
        st.info("Fill in patient data on the left and click Analyse.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  "89.00%")
        m2.metric("Recall",    "85.13%")
        m3.metric("F1 Score",  "80.63%")

st.markdown("---")
st.caption("⚕ For clinical decision support only. Not a substitute for professional medical judgment.")
