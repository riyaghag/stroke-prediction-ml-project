import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("data/stroke_data.csv")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("stroke_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #FF4B4B;
}
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
}
.stTextInput, .stNumberInput, .stSelectbox {
    background-color: #1c1f26;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🫀 AI + Doctor Stroke Risk Assessment System")
st.markdown("### Intelligent Clinical Decision Support")

# ---------------- INPUT FORM ----------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age (years)", 0, 100, 30)
hypertension = st.sidebar.selectbox("Hypertension", [0,1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0,1])
avg_glucose = st.sidebar.number_input("Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ever_married = st.sidebar.selectbox("Married", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
residence = st.sidebar.selectbox("Residence", ["Urban","Rural"])
smoking = st.sidebar.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes"])

# ---------------- ENCODING ----------------
input_dict = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'avg_glucose_level': avg_glucose,
    'bmi': bmi,
    'gender_Male': 1 if gender=="Male" else 0,
    'gender_Other': 0,
    'ever_married_Yes': 1 if ever_married=="Yes" else 0,
    'work_type_Never_worked': 1 if work_type=="Never_worked" else 0,
    'work_type_Private': 1 if work_type=="Private" else 0,
    'work_type_Self-employed': 1 if work_type=="Self-employed" else 0,
    'work_type_children': 1 if work_type=="children" else 0,
    'Residence_type_Urban': 1 if residence=="Urban" else 0,
    'smoking_status_formerly smoked': 1 if smoking=="formerly smoked" else 0,
    'smoking_status_never smoked': 1 if smoking=="never smoked" else 0,
    'smoking_status_smokes': 1 if smoking=="smokes" else 0
}

input_data = pd.DataFrame([input_dict])

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Risk"):

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    # -------- REALISTIC MEDICAL BOOST --------
    risk_boost = 0

    if age > 60: risk_boost += 0.15
    if bmi > 30: risk_boost += 0.10
    if smoking == "smokes": risk_boost += 0.15
    if avg_glucose > 140: risk_boost += 0.10
    if hypertension == 1: risk_boost += 0.15
    if heart_disease == 1: risk_boost += 0.15

    prob = min(prob + risk_boost, 1.0)

    # -------- RISK CATEGORY --------
    if prob < 0.3:
        risk_level = "Low Risk"
    elif prob < 0.7:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    # ---------------- OUTPUT ----------------
    st.markdown("## 📊 Clinical Report")

    st.write(f"### 🧠 AI Risk Probability: {prob*100:.2f}%")
    st.write(f"### ⚠️ Risk Level: {risk_level}")

    # ---------------- DOCTOR EXPLANATION ----------------
    st.markdown("## 👩‍⚕️ Doctor Insights")

    explanation = []

    if age > 60:
        explanation.append("Advanced age increases stroke risk")
    if bmi > 30:
        explanation.append("Obesity is a significant risk factor")
    if smoking == "smokes":
        explanation.append("Smoking damages blood vessels")
    if avg_glucose > 140:
        explanation.append("High glucose indicates diabetes risk")
    if hypertension == 1:
        explanation.append("Hypertension is a major cause of stroke")
    if heart_disease == 1:
        explanation.append("Heart disease increases stroke chances")

    if explanation:
        for e in explanation:
            st.write("•", e)
    else:
        st.write("No major clinical risks detected.")

    # ---------------- FINAL RECOMMENDATION ----------------
    st.markdown("## 🏥 Recommendation")

    if risk_level == "High Risk":
        st.error("Immediate medical consultation required")
    elif risk_level == "Moderate Risk":
        st.warning("Monitor health and improve lifestyle")
    else:
        st.success("Maintain healthy lifestyle")

    # ---------------- GRAPH SECTION ----------------
    st.markdown("## 📈 Risk Visualization")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.barh(["Stroke Risk"], [prob])
        ax.set_xlim(0,1)
        ax.set_title("Risk Probability")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(["Age","BMI","Glucose"], [age,bmi,avg_glucose])
        ax2.set_title("Health Metrics")
        st.pyplot(fig2)

    # ---------------- DISCLAIMER ----------------
    st.markdown("---")
    st.caption("⚠️ This system is for educational purposes only and not a medical diagnosis.")