import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('stroke_prediction_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# --- SIDEBAR: Project Insights ---
st.sidebar.title("Project Analytics")
st.sidebar.info("Based on EDA, Age is the strongest predictor (Correlation: 0.25).")
st.sidebar.write("Model Confidence (AUC): 0.79")

# --- MAIN UI ---
st.title("🧠 Professional Stroke Risk Analysis")
st.write("Enter clinical parameters below for a probability-based assessment.")

# Use a form to prevent flickering
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 0, 120, 55)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        married = st.selectbox("Ever Married", ["Yes", "No"])
        
    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1], help="0: No, 1: Yes")
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
    with col3:
        avg_glucose = st.number_input("Avg Glucose Level", value=105.0)
        bmi = st.number_input("BMI", value=28.0)
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])

    submit = st.form_submit_button("Run Diagnostic Prediction")

# --- PREDICTION & VISUALIZATION ---
if submit:
    if model and scaler:
        # 1. Prepare Features (Exact 17 features from your Scaler)
        expected = scaler.feature_names_in_
        inputs = {
            'id': 0, 'age': float(age), 'hypertension': int(hypertension),
            'heart_disease': int(heart_disease), 'avg_glucose_level': float(avg_glucose),
            'bmi': float(bmi), 'gender_Male': 1 if gender == "Male" else 0,
            'gender_Other': 1 if gender == "Other" else 0,
            'ever_married_Yes': 1 if married == "Yes" else 0,
            'work_type_Never_worked': 0, 'work_type_Private': 0, 
            'work_type_Self-employed': 0, 'work_type_children': 0,
            'Residence_type_Urban': 1 if residence == "Urban" else 0, 
            'smoking_status_formerly smoked': 1 if smoking == "formerly smoked" else 0,
            'smoking_status_never smoked': 1 if smoking == "never smoked" else 0,
            'smoking_status_smokes': 1 if smoking == "smokes" else 0
        }
        
        df_input = pd.DataFrame([inputs]).reindex(columns=expected, fill_value=0)
        
        # 2. Scaling and Probabilities
        X_scaled = scaler.transform(df_input)
        prob = model.predict_proba(X_scaled)[0][1]

        # 3. Display Results
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prob > 0.25: # Medical safety threshold
                st.error(f"### Result: HIGH RISK")
                st.write(f"**Stroke Probability:** {prob:.1%}")
                st.progress(prob)
                st.warning("This profile aligns with high-risk clinical patterns observed in the training data.")
            else:
                st.success(f"### Result: LOW RISK")
                st.write(f"**Stroke Probability:** {prob:.1%}")
                st.progress(prob)

        with res_col2:
            # Gauge-style Chart
            fig, ax = plt.subplots(figsize=(5, 2))
            colors = ['#2ecc71', '#e74c3c']
            sns.barplot(x=[prob, 1-prob], y=["Risk", "Safe"], palette=colors, ax=ax)
            ax.set_title("Probability Distribution")
            st.pyplot(fig)
            
    else:
        st.error("Model files are missing. Please upload the .pkl files to GitHub.")
