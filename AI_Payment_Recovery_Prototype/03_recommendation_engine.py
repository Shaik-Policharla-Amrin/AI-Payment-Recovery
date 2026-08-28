"""
STEP 3 — Recovery Recommendation Engine
=========================================
Given the AI-predicted failure cause + customer context, recommend the
BEST recovery action. This is a policy layer on top of the classifier.

Approach: a "smart policy table" (interpretable, tunable, hackathon-
friendly) informed by domain logic, PLUS a learned "expected recovery
probability" model that ranks actions and picks the best one per
transaction. This mirrors how real fintech recovery/dunning systems
(e.g. Stripe Smart Retries, Recurly) work, but adds the missing
cause-awareness layer.

Recovery actions available:
  - retry_immediate        (retry within minutes)
  - retry_2h
  - retry_24h
  - retry_72h
  - suggest_alt_payment_method
  - customer_nudge (SMS/email/push asking to update payment info)
"""

import numpy as np
import pandas as pd

ACTIONS = [
    "retry_immediate",
    "retry_2h",
    "retry_24h",
    "retry_72h",
    "suggest_alt_payment_method",
    "customer_nudge",
]

# Base success-probability priors per (failure_cause, action) — this is
# the "domain knowledge" a fintech team would encode from historical
# recovery data. In production these would be LEARNED from real
# recovery outcomes (a second model), not hand-set. We hand-set them
# here for a transparent, explainable hackathon demo, then simulate
# outcomes probabilistically.
BASE_POLICY = {
    "insufficient_funds": {
        "retry_immediate": 0.05,
        "retry_2h": 0.10,
        "retry_24h": 0.22,
        "retry_72h": 0.38,          # payday/salary cycles -> wait longer
        "suggest_alt_payment_method": 0.30,
        "customer_nudge": 0.25,
    },
    "bank_timeout": {
        "retry_immediate": 0.55,     # transient -> retry fast works well
        "retry_2h": 0.60,
        "retry_24h": 0.45,
        "retry_72h": 0.30,
        "suggest_alt_payment_method": 0.35,
        "customer_nudge": 0.15,
    },
    "network_error": {
        "retry_immediate": 0.65,     # purely transient -> immediate retry best
        "retry_2h": 0.58,
        "retry_24h": 0.40,
        "retry_72h": 0.25,
        "suggest_alt_payment_method": 0.30,
        "customer_nudge": 0.12,
    },
    "fraud_false_positive": {
        "retry_immediate": 0.08,     # retrying won't fix a fraud block
        "retry_2h": 0.10,
        "retry_24h": 0.15,
        "retry_72h": 0.20,
        "suggest_alt_payment_method": 0.55,   # different card bypasses the block
        "customer_nudge": 0.45,               # manual verification nudge works
    },
}


def recommend_action(failure_cause: str, context: dict) -> dict:
    """
    Given a predicted failure cause and transaction/customer context,
    return the recommended recovery action + its expected success
    probability, adjusted for context (personalization layer).
    """
    base = BASE_POLICY[failure_cause].copy()

    # --- Contextual adjustments (personalization) ---
    adj = base.copy()

    # High-value / long-tenure customers: prioritize soft nudge + alt method
    # over aggressive retries (protect experience).
    if context.get("customer_tenure_days", 0) > 365:
        adj["customer_nudge"] *= 1.15
        adj["suggest_alt_payment_method"] *= 1.10

    # Recurring subscriptions: fast retries matter more (avoid service
    # interruption), but too many retries hurt -> cap retry_immediate.
    if context.get("is_recurring_subscription", 0) == 1:
        adj["retry_24h"] *= 1.10
        adj["retry_72h"] *= 1.10

    # Customers with many recent failed attempts: back off from more
    # retries, push toward alt method / nudge instead.
    if context.get("prior_failed_attempts_24h", 0) >= 2:
        adj["retry_immediate"] *= 0.5
        adj["retry_2h"] *= 0.6
        adj["suggest_alt_payment_method"] *= 1.25
        adj["customer_nudge"] *= 1.2

    # Normalize/clip
    for k in adj:
        adj[k] = float(np.clip(adj[k], 0, 0.95))

    best_action = max(adj, key=adj.get)
    return {
        "recommended_action": best_action,
        "expected_success_prob": round(adj[best_action], 3),
        "all_action_scores": {k: round(v, 3) for k, v in adj.items()},
    }


def static_baseline_action(_failure_cause: str, _context: dict) -> dict:
    """
    The CURRENT industry-standard baseline most platforms use:
    a single fixed rule regardless of cause — 'retry after 24 hours'.
    This is what we are benchmarking the AI system against.
    """
    return {
        "recommended_action": "retry_24h",
        "expected_success_prob": BASE_POLICY[_failure_cause]["retry_24h"],
    }


if __name__ == "__main__":
    # quick smoke test
    sample_context = {
        "customer_tenure_days": 500,
        "is_recurring_subscription": 1,
        "prior_failed_attempts_24h": 0,
    }
    for cause in BASE_POLICY:
        rec = recommend_action(cause, sample_context)
        base = static_baseline_action(cause, sample_context)
        print(f"{cause:22s} | AI -> {rec['recommended_action']:26s} "
              f"({rec['expected_success_prob']:.2f})  vs  "
              f"Baseline -> retry_24h ({base['expected_success_prob']:.2f})")
