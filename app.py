import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    try:
        # Matches the filename in your VS Code screenshot
        model = pickle.load(open('stroke_prediction_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, scaler = load_models()

# --- UI HEADER ---
st.title("Stroke Risk Prediction System")
st.write("Enter patient details to predict the risk of stroke.")

# --- INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
        heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
        avg_glucose = st.number_input("Average Glucose Level", value=100.0)
        
    with col2:
        bmi = st.number_input("BMI", value=25.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        married = st.selectbox("Ever Married", ["Yes", "No"])
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- PREDICTION LOGIC ---
if st.button("Predict Stroke Risk"):
    if model is not None and scaler is not None:
        
        # 1. THE COMPLETE FEATURE LIST (Required for Shape matching)
        all_columns = [
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Female', 'gender_Male', 'gender_Other', 
            'ever_married_No', 'ever_married_Yes', 
            'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Rural', 'Residence_type_Urban',
            'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
        ]

        # 2. Create a base DataFrame with zeros
        input_df = pd.DataFrame([[0.0] * len(all_columns)], columns=all_columns)

        # 3. Fill Numerical Values
        input_df.at[0, 'age'] = float(age)
        input_df.at[0, 'hypertension'] = int(hypertension)
        input_df.at[0, 'heart_disease'] = int(heart_disease)
        input_df.at[0, 'avg_glucose_level'] = float(avg_glucose)
        input_df.at[0, 'bmi'] = float(bmi)

        # 4. Fill Categorical (One-Hot Encoding)
        # Gender
        if gender == "Male": input_df.at[0, 'gender_Male'] = 1
        elif gender == "Female": input_df.at[0, 'gender_Female'] = 1
        else: input_df.at[0, 'gender_Other'] = 1
        
        # Marriage
        if married == "Yes": input_df.at[0, 'ever_married_Yes'] = 1
        else: input_df.at[0, 'ever_married_No'] = 1
        
        # Smoking Status
        if smoking == "formerly smoked": input_df.at[0, 'smoking_status_formerly smoked'] = 1
        elif smoking == "never smoked": input_df.at[0, 'smoking_status_never smoked'] = 1
        elif smoking == "smokes": input_df.at[0, 'smoking_status_smokes'] = 1
        else: input_df.at[0, 'smoking_status_Unknown'] = 1

        # NOTE: work_type and Residence_type remain 0 (neutral) to maintain feature count

        # 5. Transform and Predict
        try:
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Get prediction
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                st.error("🚨 High Risk of Stroke")
            else:
                st.success("✅ Low Risk of Stroke")
