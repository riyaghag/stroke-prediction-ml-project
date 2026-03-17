import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

st.title("Stroke Risk Prediction System")
st.write("This app predicts the risk of stroke based on health parameters.")

st.sidebar.title("Stroke Prediction App")
st.sidebar.info("Enter patient health details to check stroke risk.")

# ---------------- LOAD DATASET SAFELY ----------------
try:
    df = pd.read_csv("data/stroke_data.csv")
except:
    st.warning("Dataset could not be loaded.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("stroke_prediction_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# ---------------- USER INPUTS ----------------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    avg_glucose_level = st.number_input("Average Glucose Level")

with col2:
    bmi = st.number_input("BMI")
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked","formerly smoked","smokes","Unknown"])

# ---------------- SIMPLE ENCODING ----------------
gender = 1 if gender == "Male" else 0
ever_married = 1 if ever_married == "Yes" else 0

smoking_map = {
    "never smoked":0,
    "formerly smoked":1,
    "smokes":2,
    "Unknown":3
}

smoking_status = smoking_map[smoking_status]

# ---------------- PREDICTION ----------------
if st.button("Predict Stroke Risk"):

    input_data = np.array([[age, hypertension, heart_disease,
                            avg_glucose_level, bmi,
                            gender, ever_married, smoking_status]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

# ---------------- SHOW VISUALIZATIONS ----------------
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.image("confusion_matrix.png", caption="Confusion Matrix")

with col2:
    st.image("roc_curve.png", caption="ROC Curve")

st.image("feature_importance.png", caption="Feature Importance")
