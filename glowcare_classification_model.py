# ==========================
# IMPORT LIBRARIES
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_excel("GlowCare_Customer_Data.xlsx")

print(df.head())

# ==========================
# DATASET INFO
# ==========================

print(df.info())

# ==========================
# CHECK MISSING VALUES
# ==========================

print(df.isnull().sum())

# ==========================
# ENCODING
# ==========================

encoder = LabelEncoder()

df["Income_Level"] = encoder.fit_transform(df["Income_Level"])

print(df.head())

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop("Will_Buy_Serum", axis=1)
y = df["Will_Buy_Serum"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# LOGISTIC REGRESSION
# ==========================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n Logistic Regression Results")

print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall   :", recall_score(y_test, y_pred_log))
print("F1 Score :", f1_score(y_test, y_pred_log))

# ==========================
# RANDOM FOREST
# ==========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n Random Forest Results")

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))

# ==========================
# CLASSIFICATION REPORT
# ==========================

print("\n Classification Report\n")

print(classification_report(y_test, y_pred_rf))

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

# ==========================
# FEATURE IMPORTANCE PLOT
# ==========================

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Feature Importance")

plt.show()

# ==========================
# SAMPLE PREDICTION
# ==========================

sample_customer = pd.DataFrame({
    "Age": [28],
    "Income_Level": [2],
    "Skincare_Interest_Score": [78],
    "Daily_App_Time_Mins": [25],
    "Previous_Purchases": [6]
})

prediction = rf_model.predict(sample_customer)

probability = rf_model.predict_proba(sample_customer)

print("\n Prediction Result")

if prediction[0] == 1:
    print("Customer WILL buy the serum")
else:
    print("Customer will NOT buy the serum")

print("Probability:", probability)

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(rf_model, "GlowCare_RF_Model.pkl")

print("\n Model Saved Successfully")