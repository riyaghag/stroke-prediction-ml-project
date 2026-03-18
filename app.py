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
        # Using the filenames from your VS Code screenshot
        model = pickle.load(open('stroke_prediction_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, scaler = load_models()

# --- UI ---
st.title("Stroke Risk Prediction System")
st.write("Enter patient details to predict risk.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        avg_glucose = st.number_input("Avg Glucose Level", value=100.0)
    with col2:
        bmi = st.number_input("BMI", value=25.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        married = st.selectbox("Ever Married", ["Yes", "No"])
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- PREDICTION ---
if st.button("Predict Stroke Risk"):
    if model and scaler:
        try:
            # 1. THE SAFETY TRICK: Ask the scaler exactly what it wants
            # This handles the 17 vs 21 feature mismatch automatically
            expected_features = scaler.feature_names_in_
            
            # 2. Build a dictionary for all possible inputs
            input_dict = {
                'id': 0, # Scaler asked for 'id' in your last error
                'age': float(age),
                'hypertension': int(hypertension),
                'heart_disease': int(heart_disease),
                'avg_glucose_level': float(avg_glucose),
                'bmi': float(bmi),
                'gender_Male': 1 if gender == "Male" else 0,
                'gender_Other': 1 if gender == "Other" else 0,
                'ever_married_Yes': 1 if married == "Yes" else 0,
                'work_type_Never_worked': 0, 
                'work_type_Private': 0, 
                'work_type_Self-employed': 0, 
                'work_type_children': 0,
                'Residence_type_Urban': 0, 
                'smoking_status_formerly smoked': 1 if smoking == "formerly smoked" else 0,
                'smoking_status_never smoked': 1 if smoking == "never smoked" else 0,
                'smoking_status_smokes': 1 if smoking == "smokes" else 0
            }

            # 3. Create DataFrame and FORCE it to match the scaler's order
            # This prevents the "Feature names unseen at fit time" error
            df = pd.DataFrame([input_dict])
            df = df.reindex(columns=expected_features, fill_value=0)

            # 4. Transform and Predict
            X_scaled = scaler.transform(df) 
            prediction = model.predict(X_scaled)
            
            if prediction[0] == 1:
                st.error("🚨 High Risk of Stroke")
            else:
                st.success("✅ Low Risk of Stroke")
                
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            # If it still fails, this will show us exactly why
            st.write("Expected by scaler:", list(scaler.feature_names_in_))
    else:
        st.warning("Files missing. Please check GitHub.")
