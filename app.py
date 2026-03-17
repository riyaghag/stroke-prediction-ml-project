import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Stroke Prediction System", layout="wide")

# Load model safely
try:
    model = pickle.load(open("stroke_prediction_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Title
st.title("🧠 Stroke Risk Prediction Dashboard")
st.markdown("### Clinical Decision Support System")
st.info("⚠️ This tool is for educational purposes and should not replace medical advice.")

# Sidebar Inputs
st.sidebar.header("🧾 Patient Details")

age = st.sidebar.number_input("Age (years)", 0, 120)
hypertension = st.sidebar.selectbox("Hypertension", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
heart_disease = st.sidebar.selectbox("Heart Disease", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

avg_glucose = st.sidebar.number_input("Glucose Level (mg/dL)", 50.0, 300.0)
bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 60.0)

gender = st.sidebar.selectbox("Gender", ["Female","Male","Other"])
married = st.sidebar.selectbox("Married", ["No","Yes"])
work = st.sidebar.selectbox("Work Type", ["Private","Self-employed","Govt_job","children","Never_worked"])
residence = st.sidebar.selectbox("Residence", ["Rural","Urban"])
smoking = st.sidebar.selectbox("Smoking", ["Unknown","formerly smoked","never smoked","smokes"])

# Patient Summary
st.markdown("## 🧾 Patient Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"Age: {age} yrs")
    st.info(f"BMI: {bmi}")

with col2:
    st.info(f"Glucose: {avg_glucose} mg/dL")
    st.info(f"Hypertension: {'Yes' if hypertension else 'No'}")

with col3:
    st.info(f"Heart Disease: {'Yes' if heart_disease else 'No'}")
    st.info(f"Smoking: {smoking}")

# Prediction
st.markdown("## 📊 Clinical Risk Analysis")

if st.button("🔍 Analyze Patient Risk"):

    features = np.zeros(17)

    # Numerical
    features[0] = age
    features[1] = hypertension
    features[2] = heart_disease
    features[3] = avg_glucose
    features[4] = bmi

    # Encoding
    if gender == "Male": features[5] = 1
    elif gender == "Other": features[6] = 1

    if married == "Yes": features[7] = 1

    if work == "Never_worked": features[8] = 1
    elif work == "Private": features[9] = 1
    elif work == "Self-employed": features[10] = 1
    elif work == "children": features[11] = 1

    if residence == "Urban": features[12] = 1

    if smoking == "formerly smoked": features[13] = 1
    elif smoking == "never smoked": features[14] = 1
    elif smoking == "smokes": features[15] = 1

    input_scaled = scaler.transform(features.reshape(1,-1))

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📊 Risk Score")
        st.progress(float(prob))
        st.metric("Stroke Probability", f"{prob*100:.2f}%")

        if prob < 0.3:
            st.success("🟢 Low Risk Zone")
        elif prob < 0.7:
            st.warning("🟠 Moderate Risk Zone")
        else:
            st.error("🔴 High Risk Zone")

    with colB:
        st.subheader("🩺 Diagnosis")
        if prediction[0] == 1:
            st.error("⚠️ High Stroke Risk")
        else:
            st.success("✅ Low Stroke Risk")

    st.markdown("## 🧠 Clinical Interpretation")

    if prob > 0.7:
        st.warning("🔴 Critical Risk — Immediate medical evaluation required.")
    elif prob > 0.4:
        st.warning("🟠 Moderate Risk — Lifestyle and monitoring recommended.")
    else:
        st.info("🟢 Low Risk — Maintain healthy habits.")

    st.markdown("## 🏥 Clinical Indicators")

    if bmi > 30:
        st.warning("⚠️ Obesity detected")

    if avg_glucose > 140:
        st.warning("⚠️ High glucose level")

    if age > 60:
        st.warning("⚠️ Age-related risk factor")

# Footer
st.markdown("---")
st.markdown("🧬 Developed using Machine Learning | Healthcare AI System")