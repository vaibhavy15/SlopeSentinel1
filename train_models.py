import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("rockfall_dataset.csv")

X = data.drop("risk_level", axis=1)
y = data["risk_level"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scale Data (For Logistic Regression)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 1️⃣ Logistic Regression
# =============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

log_acc = accuracy_score(y_test, log_pred)

print("\n=== Logistic Regression ===")
print("Accuracy:", log_acc)
print(classification_report(y_test, log_pred))

# =============================
# 2️⃣ Random Forest
# =============================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("\n=== Random Forest ===")
print("Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

# =============================
# 3️⃣ XGBoost
# =============================
xgb_model = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)

print("\n=== XGBoost ===")
print("Accuracy:", xgb_acc)
print(classification_report(y_test, xgb_pred))

# -----------------------------
# Model Comparison
# -----------------------------
print("\n=== Model Comparison ===")
print("Logistic Regression:", log_acc)
print("Random Forest:", rf_acc)
print("XGBoost:", xgb_acc)

# -----------------------------
# Cross Validation (5-Fold)
# -----------------------------
print("\n=== 5-Fold Cross Validation ===")

log_cv = cross_val_score(
    LogisticRegression(max_iter=1000),
    scaler.fit_transform(X),
    y,
    cv=5
)

rf_cv = cross_val_score(
    RandomForestClassifier(n_estimators=300, random_state=42),
    X,
    y,
    cv=5
)

xgb_cv = cross_val_score(
    XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        random_state=42
    ),
    X,
    y,
    cv=5
)

print("Logistic Regression CV Accuracy:", log_cv.mean())
print("Random Forest CV Accuracy:", rf_cv.mean())
print("XGBoost CV Accuracy:", xgb_cv.mean())

# -----------------------------
# Confusion Matrix (Best Model)
# -----------------------------
best_model = log_model
best_pred = log_pred

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Feature Importance (Random Forest)
# -----------------------------
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.show()

# -----------------------------
# Save Best Model
# -----------------------------

best_model = log_model   # since Logistic Regression performed best

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Best model and scaler saved successfully!")

