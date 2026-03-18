# 1. THE REVISED 17-FEATURE LIST (Matched to your Scaler)
all_columns = [
    'id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', 
    'ever_married_Yes', 
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'Residence_type_Urban', 
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
]

# 2. Create the template DataFrame
df = pd.DataFrame([[0.0] * len(all_columns)], columns=all_columns)

# 3. Fill 'id' with a dummy value (since it's required but doesn't affect health)
df.at[0, 'id'] = 0.0

# 4. Fill Numerical Values
df.at[0, 'age'] = float(age)
df.at[0, 'hypertension'] = int(hypertension)
df.at[0, 'heart_disease'] = int(heart_disease)
df.at[0, 'avg_glucose_level'] = float(avg_glucose)
df.at[0, 'bmi'] = float(bmi)

# 5. Fill Categorical Logic (Removed the 'No' and 'Female' columns the scaler hates)
if gender == "Male": df.at[0, 'gender_Male'] = 1
elif gender == "Other": df.at[0, 'gender_Other'] = 1

if married == "Yes": df.at[0, 'ever_married_Yes'] = 1

if smoking == "formerly smoked": df.at[0, 'smoking_status_formerly smoked'] = 1
elif smoking == "never smoked": df.at[0, 'smoking_status_never smoked'] = 1
elif smoking == "smokes": df.at[0, 'smoking_status_smokes'] = 1
