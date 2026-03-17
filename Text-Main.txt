import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

print(df.head())
print(df.describe())

print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nData types:\n")
print(df.info())
print("\nMissing values:\n")
print(df.isnull().sum())

print("\nStroke Value Counts:\n")
print(df["stroke"].value_counts())
print("\nStroke Percentage:\n")
print(df["stroke"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt

df["stroke"].value_counts().plot(kind="bar")
plt.title("Stroke Distribution")
plt.xlabel("Stroke")
plt.ylabel("Count")
plt.show()

df["age"].plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

print("\nMissing values in BMI:")
print(df["bmi"].isnull().sum())

df["bmi"].plot(kind="hist", bins=20)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

print("Missing BMI after filling:", df["bmi"].isnull().sum())
print("\nStroke distribution:")
print(df["stroke"].value_counts())

df.boxplot(column="age", by="stroke")   
plt.title("Age vs Stroke")
plt.suptitle("")
plt.xlabel("Stroke")
plt.ylabel("Age")
plt.show()                    #Stroke cases are concentrated among older individuals.

#hypertension vs stroke
print("\nStroke rate by hypertension:")
print(df.groupby("hypertension")["stroke"].mean())

df.groupby("hypertension")["stroke"].mean().plot(kind="bar")
plt.title("Stroke Rate by Hypertension")
plt.xlabel("Hypertension")
plt.ylabel("Stroke Rate")
plt.show()

#heart_disease vs stroke
print("\nStroke rate by hypertension:")
print(df.groupby("hypertension")["stroke"].mean())

df.groupby("hypertension")["stroke"].mean().plot(kind="bar")
plt.title("Stroke Rate by Hypertension")
plt.xlabel("Hypertension")
plt.ylabel("Stroke Rate")
plt.show()

#Analyzing other factors  
df["avg_glucose_level"].plot(kind="hist", bins=30)  #GLUCOSE LEVEL DISTRIBUTION
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.show()

df.boxplot(column="bmi", by="stroke")          #BMI VS STROKE
plt.title("BMI vs Stroke")
plt.suptitle("")
plt.xlabel("Stroke")
plt.ylabel("BMI")
plt.show()

print(df.corr(numeric_only=True))

#correation analysis
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

df_encoded = pd.get_dummies(df, drop_first=True) # Converting categorical variables to numeric

print("\nEncoded Columns:")
print(df_encoded.columns)

X = df_encoded.drop("stroke", axis=1)  #defining features and target
y = df_encoded["stroke"]

from sklearn.model_selection import train_test_split  #train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y                 #using stratify keeps the stroke proportion balanced in both train and test sets
)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

from sklearn.preprocessing import StandardScaler    #feature scaling using standard scaler to normalize the data
scaler = StandardScaler()

X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression #logistic regression model training

model = LogisticRegression(max_iter=1000)
model.fit(X_train_smote, y_train_smote)    

y_pred = model.predict(X_test) #predictions

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred)) #model evaluation

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

plt.title("Confusion Matrix")  #confusion matrix visualization
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

#ROC curve and AUC calculation
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right")
plt.show()

#Random Forest which is a tree model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train_smote, y_train_smote)

rf_pred = rf_model.predict(X_test)

print("Random Forest Results:\n")
print(classification_report(y_test, rf_pred))

from sklearn.metrics import roc_curve, auc   #adding ROC and AUC for RF

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

importance = rf_model.feature_importances_   #adding feature importance

features = X.columns

importance_df = pd.DataFrame({
    "Feature":features,
    "Importance":importance
}).sort_values(by="Importance", ascending=False)

sns.barplot(x="Importance", y="Feature", data=importance_df)

plt.title("Feature Importance")
plt.show()

#XGBoost model training
from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb_model.fit(X_train_smote, y_train_smote)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results:\n")
print(classification_report(y_test, xgb_pred))

# Probabilities for ROC curves
lr_prob = model.predict_proba(X_test)[:,1]
rf_prob = rf_model.predict_proba(X_test)[:,1]
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

# ROC calculations
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

# AUC scores
lr_auc = auc(lr_fpr, lr_tpr)
rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

# Plot
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.2f})")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Comparison ROC Curve")
plt.legend()

plt.show()

# Model comparison table using accuracy scores
from sklearn.metrics import accuracy_score

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred)
    ]
})

print("\nModel Comparison:")
print(results)


# Cross-validation to check model reliability
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_model, X_train_smote, y_train_smote, cv=5)

print("\nRandom Forest Cross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

with open("stroke_prediction_model.pkl", "wb") as file:  # Save trained Random Forest model
    pickle.dump(rf_model, file)                  
print("Model saved successfully!")

with open("scaler.pkl", "wb") as file:  # Save scaled data
    pickle.dump(scaler, file)
print("Scaler saved successfully!")

importance_df.to_csv("feature_importance.csv", index=False) # Save feature importance file
print("Feature importance saved.")
