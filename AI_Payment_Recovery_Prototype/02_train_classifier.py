"""
STEP 2 — AI Failure Classifier
===============================
This is the core AI model: given the observable signals of a failed
transaction, predict the REAL underlying cause of failure.

Model: RandomForestClassifier (fast, interpretable via feature
importance, works well on tabular fintech data — good hackathon choice
because you can explain it in 30 seconds to judges).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("/home/claude/payment_recovery/failed_transactions.csv")

FEATURES = [
    "amount",
    "bank_response_code",
    "network_latency_ms",
    "fraud_score",
    "prior_failed_attempts_24h",
    "customer_tenure_days",
    "account_balance_signal",
    "hour_of_day",
    "is_recurring_subscription",
]
CATEGORICAL = ["payment_method"]

# One-hot encode payment_method
df_encoded = pd.get_dummies(df, columns=CATEGORICAL, prefix="pm")
feature_cols = FEATURES + [c for c in df_encoded.columns if c.startswith("pm_")]

X = df_encoded[feature_cols]
y = df_encoded["true_failure_cause"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc:.3f}\n")
print("Classification Report:\n", report)

# Feature importance
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Top feature importances:\n", importances.head(8))

# Save everything for the next stages
joblib.dump(clf, "/home/claude/payment_recovery/failure_classifier.pkl")
joblib.dump(le, "/home/claude/payment_recovery/label_encoder.pkl")
joblib.dump(feature_cols, "/home/claude/payment_recovery/feature_cols.pkl")
np.save("/home/claude/payment_recovery/confusion_matrix.npy", cm)
importances.to_csv("/home/claude/payment_recovery/feature_importances.csv")

with open("/home/claude/payment_recovery/metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {acc:.3f}\n\n")
    f.write(report)

print("\nSaved model, encoder, and metrics.")
