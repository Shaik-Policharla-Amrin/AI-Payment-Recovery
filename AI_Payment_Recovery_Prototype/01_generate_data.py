"""
STEP 1 — Synthetic Data Generation
==================================
Simulates a realistic stream of FAILED digital payment transactions.
Each failed transaction has a hidden "true cause" and observable signals
that a real payment gateway would log (bank response code, network
latency, fraud score, customer history, etc).

In production, this data would come from your payment gateway logs
(Stripe, Razorpay, PayU, bank webhooks, etc). Here we simulate it so
the whole pipeline is reproducible and demo-able.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000  # number of failed transactions to simulate

FAILURE_CAUSES = [
    "insufficient_funds",
    "bank_timeout",
    "network_error",
    "fraud_false_positive",
]

# Ground-truth cause distribution (realistic-ish skew)
cause_probs = [0.40, 0.22, 0.23, 0.15]

rows = []
for i in range(N):
    cause = np.random.choice(FAILURE_CAUSES, p=cause_probs)

    # --- Feature generation, correlated with the true cause ---
    if cause == "insufficient_funds":
        amount = np.random.gamma(shape=2.0, scale=1800)          # larger amounts
        bank_response_code = np.random.choice([51, 61, 65], p=[0.7, 0.2, 0.1])
        network_latency_ms = np.random.normal(180, 40)
        fraud_score = np.random.beta(2, 8)                       # low fraud score
        prior_failed_attempts_24h = np.random.poisson(1.2)
        customer_tenure_days = np.random.exponential(400)
        account_balance_signal = np.random.beta(2, 6)            # low balance signal
    elif cause == "bank_timeout":
        amount = np.random.gamma(shape=2.0, scale=1000)
        bank_response_code = np.random.choice([91, 96, 68], p=[0.5, 0.3, 0.2])
        network_latency_ms = np.random.normal(350, 80)
        fraud_score = np.random.beta(2, 8)
        prior_failed_attempts_24h = np.random.poisson(0.6)
        customer_tenure_days = np.random.exponential(500)
        account_balance_signal = np.random.beta(4, 4)
    elif cause == "network_error":
        amount = np.random.gamma(shape=2.0, scale=900)
        bank_response_code = np.random.choice([96, 21, 5], p=[0.5, 0.3, 0.2])
        network_latency_ms = np.random.normal(500, 120)          # highest latency
        fraud_score = np.random.beta(2, 8)
        prior_failed_attempts_24h = np.random.poisson(0.4)
        customer_tenure_days = np.random.exponential(500)
        account_balance_signal = np.random.beta(4, 4)
    else:  # fraud_false_positive
        amount = np.random.gamma(shape=2.0, scale=2500)          # unusually large/odd amounts
        bank_response_code = np.random.choice([59, 63, 5], p=[0.6, 0.2, 0.2])
        network_latency_ms = np.random.normal(200, 50)
        fraud_score = np.random.beta(8, 3)                       # high fraud score (but false)
        prior_failed_attempts_24h = np.random.poisson(0.3)
        customer_tenure_days = np.random.exponential(900)        # often long-standing customers
        account_balance_signal = np.random.beta(6, 3)

    payment_method = np.random.choice(
        ["credit_card", "debit_card", "upi", "netbanking", "wallet"],
        p=[0.30, 0.30, 0.20, 0.12, 0.08],
    )
    hour_of_day = np.random.randint(0, 24)
    is_recurring_subscription = np.random.choice([0, 1], p=[0.65, 0.35])

    rows.append(
        {
            "transaction_id": f"TXN{100000+i}",
            "amount": round(max(amount, 10), 2),
            "payment_method": payment_method,
            "bank_response_code": bank_response_code,
            "network_latency_ms": max(round(network_latency_ms, 1), 5),
            "fraud_score": round(np.clip(fraud_score, 0, 1), 3),
            "prior_failed_attempts_24h": prior_failed_attempts_24h,
            "customer_tenure_days": round(max(customer_tenure_days, 1)),
            "account_balance_signal": round(np.clip(account_balance_signal, 0, 1), 3),
            "hour_of_day": hour_of_day,
            "is_recurring_subscription": is_recurring_subscription,
            "true_failure_cause": cause,  # label (would come from reconciliation/bank data historically)
        }
    )

df = pd.DataFrame(rows)
df.to_csv("/home/claude/payment_recovery/failed_transactions.csv", index=False)
print(f"Generated {len(df)} synthetic failed transactions.")
print(df["true_failure_cause"].value_counts(normalize=True).round(3))
print(df.head())
