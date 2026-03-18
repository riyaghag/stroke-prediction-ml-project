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
        # Matches your filename from the screenshot
        model = pickle.load(open('stroke_prediction_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, scaler = load_models()

# --- UI ---
st.title("Stroke Risk Prediction System")
st.write("Fill in the details below to predict stroke risk.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        hypertension = st.selectbox("Hypertension", [0, 1], help="0 = No, 1 = Yes")
        heart_disease = st.selectbox("Heart Disease", [0, 1], help="0 = No, 1 = Yes")
        avg_glucose = st.number_input("Avg Glucose Level", value=100.0)
    with col2:
        bmi = st.number_input("BMI", value=25.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        married = st.selectbox("Ever Married", ["Yes", "No"])
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- PREDICTION ---
if st.button("Predict Stroke Risk"):
    if model and scaler:
        # 1. COMPLETE 21-FEATURE LIST (Matches standard Kaggle preprocessing)
        all_columns = [
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Female', 'gender_Male', 'gender_Other', 
            'ever_married_No', 'ever_married_Yes', 
            'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Rural', 'Residence_type_Urban',
            'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
        ]
        
        # 2. Create the template DataFrame with zeros
        df = pd.DataFrame([[0.0] * len(all_columns)], columns=all_columns)
        
        # 3. Fill Numerical Values
        df.at[0, 'age'] = float(age)
        df.at[0, 'hypertension'] = int(hypertension)
        df.at[0, 'heart_disease'] = int(heart_disease)
        df.at[0, 'avg_glucose_level'] = float(avg_glucose)
        df.at[0, 'bmi'] = float(bmi)
        
        # 4. Fill Categorical Logic (One-Hot Encoding)
        if gender == "Male": df.at[0, 'gender_Male'] = 1
        elif gender == "Female": df.at[0, 'gender_Female'] = 1
        else: df.at[0, 'gender_Other'] = 1

        if married == "Yes": df.at[0, 'ever_married_Yes'] = 1
        else: df.at[0, 'ever_married_No'] = 1

        if smoking == "formerly smoked": df.at[0, 'smoking_status_formerly smoked'] = 1
        elif smoking == "never smoked": df.at[0, 'smoking_status_never smoked'] = 1
        elif smoking == "smokes": df.at[0, 'smoking_status_smokes'] = 1
        else: df.at[0, 'smoking_status_Unknown'] = 1

        # 5. Transform and Predict with Error Handling
        try:
            X_scaled = scaler.transform(df) 
            prediction = model.predict(X_scaled)
            
            if prediction[0] == 1:
                st.error("🚨 High Risk of Stroke")
            else:
                st.success("✅ Low Risk of Stroke")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info(f"Scaler expects {scaler.n_features_in_} features. Code sent {len(all_columns)}.")
    else:
        st.warning("Model or Scaler not loaded properly. Check your .pkl files.")
