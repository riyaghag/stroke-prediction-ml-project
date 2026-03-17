import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Stroke Prediction App", layout="wide")

# Load model and scaler
model = pickle.load(open("stroke_prediction_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🧠 Stroke Prediction System")
st.markdown("### AI-powered risk prediction based on patient health data")

st.info("⚠️ This prediction is based on a machine learning model and should not replace medical advice.")

# Sidebar for inputs
st.sidebar.header("📋 Enter Patient Details")

# Inputs
age = st.sidebar.number_input("Age (years)", 0, 120, help="Enter age in years")

hypertension = st.sidebar.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

avg_glucose = st.sidebar.number_input(
    "Average Glucose Level (mg/dL)",
    50.0, 300.0,
    help="Normal range: 70–140 mg/dL"
)

bmi = st.sidebar.number_input(
    "BMI (kg/m²)",
    10.0, 60.0,
    help="Normal range: 18.5–24.9"
)

gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work = st.sidebar.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)
residence = st.sidebar.selectbox("Residence Type", ["Rural", "Urban"])
smoking = st.sidebar.selectbox(
    "Smoking Status",
    ["Unknown", "formerly smoked", "never smoked", "smokes"]
)

# Main area
st.subheader("📊 Prediction Result")

if st.button("Predict Stroke Risk"):

    # Create feature array
    features = np.zeros(17)

    # Numerical features
    features[0] = age
    features[1] = hypertension
    features[2] = heart_disease
    features[3] = avg_glucose
    features[4] = bmi

    # Gender
    if gender == "Male":
        features[5] = 1
    elif gender == "Other":
        features[6] = 1

    # Married
    if married == "Yes":
        features[7] = 1

    # Work type
    if work == "Never_worked":
        features[8] = 1
    elif work == "Private":
        features[9] = 1
    elif work == "Self-employed":
        features[10] = 1
    elif work == "children":
        features[11] = 1

    # Residence
    if residence == "Urban":
        features[12] = 1

    # Smoking
    if smoking == "formerly smoked":
        features[13] = 1
    elif smoking == "never smoked":
        features[14] = 1
    elif smoking == "smokes":
        features[15] = 1

    # Reshape
    input_data = features.reshape(1, -1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Stroke Probability", value=f"{probability*100:.2f}%")

    with col2:
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Stroke")
        else:
            st.success("✅ Low Risk of Stroke")

    # Extra interpretation
    if probability > 0.7:
        st.warning("🔴 Very High Risk — Immediate medical consultation recommended.")
    elif probability > 0.4:
        st.warning("🟠 Moderate Risk — Lifestyle and health monitoring advised.")
    else:
        st.info("🟢 Low Risk — Maintain healthy lifestyle.")

# Footer
st.markdown("---")
st.markdown("💡 Built using Machine Learning | Random Forest / XGBoost Model")