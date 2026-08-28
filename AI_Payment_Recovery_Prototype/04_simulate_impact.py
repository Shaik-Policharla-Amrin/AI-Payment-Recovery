"""
STEP 4 — End-to-End Simulation & Business Impact
==================================================
Runs the FULL pipeline on every failed transaction:
  raw signals -> AI classifier -> predicted cause -> recommendation
  engine -> recommended action -> simulated recovery outcome

Compares against the static baseline (fixed 24h retry for everyone)
to quantify: recovery rate lift, revenue recovered, and where the
biggest wins come from.
"""

import numpy as np
import pandas as pd
import joblib
from importlib import import_module
import sys

sys.path.insert(0, "/home/claude/payment_recovery")
from importlib import reload
import importlib.util

spec = importlib.util.spec_from_file_location(
    "rec_engine", "/home/claude/payment_recovery/03_recommendation_engine.py"
)
rec_engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec_engine)

np.random.seed(7)

df = pd.read_csv("/home/claude/payment_recovery/failed_transactions.csv")
clf = joblib.load("/home/claude/payment_recovery/failure_classifier.pkl")
le = joblib.load("/home/claude/payment_recovery/label_encoder.pkl")
feature_cols = joblib.load("/home/claude/payment_recovery/feature_cols.pkl")

df_encoded = pd.get_dummies(df, columns=["payment_method"], prefix="pm")
for c in feature_cols:
    if c not in df_encoded.columns:
        df_encoded[c] = 0
X_all = df_encoded[feature_cols]

pred_encoded = clf.predict(X_all)
df["predicted_failure_cause"] = le.inverse_transform(pred_encoded)

results = []
for _, row in df.iterrows():
    context = {
        "customer_tenure_days": row["customer_tenure_days"],
        "is_recurring_subscription": row["is_recurring_subscription"],
        "prior_failed_attempts_24h": row["prior_failed_attempts_24h"],
    }

    ai_rec = rec_engine.recommend_action(row["predicted_failure_cause"], context)
    baseline_rec = rec_engine.static_baseline_action(row["true_failure_cause"], context)

    # Simulate actual recovery outcome via Bernoulli draw using the
    # TRUE cause's real success probability for the chosen action
    # (so a wrong AI diagnosis is correctly penalized).
    true_cause = row["true_failure_cause"]
    ai_action = ai_rec["recommended_action"]
    ai_true_prob = rec_engine.BASE_POLICY[true_cause][ai_action]
    ai_recovered = np.random.rand() < ai_true_prob

    baseline_action = baseline_rec["recommended_action"]
    baseline_true_prob = rec_engine.BASE_POLICY[true_cause][baseline_action]
    baseline_recovered = np.random.rand() < baseline_true_prob

    results.append(
        {
            "transaction_id": row["transaction_id"],
            "amount": row["amount"],
            "true_failure_cause": true_cause,
            "predicted_failure_cause": row["predicted_failure_cause"],
            "ai_action": ai_action,
            "ai_recovered": ai_recovered,
            "baseline_action": baseline_action,
            "baseline_recovered": baseline_recovered,
        }
    )

res_df = pd.DataFrame(results)
res_df.to_csv("/home/claude/payment_recovery/simulation_results.csv", index=False)

# --- Business impact summary ---
total_failed_value = res_df["amount"].sum()
ai_recovered_value = res_df.loc[res_df["ai_recovered"], "amount"].sum()
baseline_recovered_value = res_df.loc[res_df["baseline_recovered"], "amount"].sum()

ai_recovery_rate = res_df["ai_recovered"].mean()
baseline_recovery_rate = res_df["baseline_recovered"].mean()

summary = {
    "total_failed_transactions": len(res_df),
    "total_failed_value": round(total_failed_value, 2),
    "ai_recovery_rate": round(ai_recovery_rate, 4),
    "baseline_recovery_rate": round(baseline_recovery_rate, 4),
    "recovery_rate_lift_pct_points": round((ai_recovery_rate - baseline_recovery_rate) * 100, 2),
    "ai_recovered_value": round(ai_recovered_value, 2),
    "baseline_recovered_value": round(baseline_recovered_value, 2),
    "incremental_revenue_recovered": round(ai_recovered_value - baseline_recovered_value, 2),
    "incremental_revenue_pct_of_total_failed": round(
        (ai_recovered_value - baseline_recovered_value) / total_failed_value * 100, 2
    ),
}

print("=== BUSINESS IMPACT SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")

pd.Series(summary).to_csv("/home/claude/payment_recovery/business_impact_summary.csv")
print("\nSaved simulation_results.csv and business_impact_summary.csv")
