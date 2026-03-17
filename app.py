import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Stroke Prediction System", layout="wide")

# CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #0b3c5d;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open("stroke_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Header
st.markdown('<div class="title">🧠 Stroke Risk Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-driven clinical decision support system</div>', unsafe_allow_html=True)

st.info("⚠️ This tool is for educational purposes and should not replace medical advice.")

# Sidebar Inputs
st.sidebar.header("🧾 Patient Information")

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

# Prediction Section
st.markdown("## 📊 Clinical Risk Analysis")

if st.button("🔍 Analyze Patient Risk"):

    features = np.zeros(17)

    # Numeric
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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Risk Score")
        st.progress(float(prob))

        st.metric("Stroke Probability", f"{prob*100:.2f}%")

    with col2:
        st.markdown("### 🩺 Diagnosis")

        if prediction[0] == 1:
            st.error("⚠️ High Stroke Risk")
        else:
            st.success("✅ Low Stroke Risk")

    # Interpretation
    st.markdown("### 🧠 Clinical Interpretation")

    if prob > 0.7:
        st.warning("🔴 Critical Risk — Immediate medical evaluation required.")
    elif prob > 0.4:
        st.warning("🟠 Moderate Risk — Lifestyle and monitoring recommended.")
    else:
        st.info("🟢 Low Risk — Maintain healthy habits.")

# Footer
st.markdown("---")
st.markdown("🧬 Developed using Machine Learning | Healthcare AI System")