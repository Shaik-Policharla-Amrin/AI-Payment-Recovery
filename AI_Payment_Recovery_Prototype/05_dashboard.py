import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import importlib.util


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Payment Recovery",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_DIR / "failed_transactions.csv"
    )

    res_df = pd.read_csv(
        RESULTS_DIR / "simulation_results.csv"
    )

    summary_df = pd.read_csv(
        RESULTS_DIR / "business_impact_summary.csv",
        index_col=0
    )

    summary = summary_df.iloc[:, 0]

    return df, res_df, summary


df, res_df, summary = load_data()


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_model():

    clf = joblib.load(
        MODELS_DIR / "failure_classifier.pkl"
    )

    le = joblib.load(
        MODELS_DIR / "label_encoder.pkl"
    )

    feature_cols = joblib.load(
        MODELS_DIR / "feature_cols.pkl"
    )

    return clf, le, feature_cols


clf, le, feature_cols = load_model()


# ============================================================
# LOAD RECOMMENDATION ENGINE
# ============================================================

@st.cache_resource
def load_recommendation_engine():

    path = ROOT / "AI_Payment_Recovery_Prototype" / "03_recommendation_engine.py"

    spec = importlib.util.spec_from_file_location(
        "rec_engine",
        path
    )

    rec_engine = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(rec_engine)

    return rec_engine


rec_engine = load_recommendation_engine()


# ============================================================
# HEADER
# ============================================================

st.title("💳 AI Payment Recovery System")

st.markdown(
    """
    **Intelligent diagnosis and recovery of failed digital payments**
    
    Instead of blindly retrying every failed payment, the system
    identifies the likely failure cause and recommends a suitable
    recovery action.
    """
)


# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3, col4 = st.columns(4)


total_transactions = int(
    summary["total_failed_transactions"]
)

ai_rate = float(
    summary["ai_recovery_rate"]
)

baseline_rate = float(
    summary["baseline_recovery_rate"]
)

ai_revenue = float(
    summary["ai_recovered_value"]
)

incremental_revenue = float(
    summary["incremental_revenue_recovered"]
)


col1.metric(
    "Failed Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "AI Recovery Rate",
    f"{ai_rate * 100:.1f}%"
)

col3.metric(
    "Baseline Recovery",
    f"{baseline_rate * 100:.1f}%"
)

def format_inr(amount):
    if amount >= 10_000_000:
        return f"₹{amount / 10_000_000:.2f} Cr"
    elif amount >= 100_000:
        return f"₹{amount / 100_000:.2f} L"
    else:
        return f"₹{amount:,.0f}"
col4.metric(
    "Additional Revenue",
    format_inr(incremental_revenue)
)


st.divider()


# ============================================================
# LIVE PAYMENT ANALYZER
# ============================================================

st.header("🔍 Analyze a Failed Payment")

st.write(
    "Enter a failed transaction and let the AI diagnose the "
    "failure and recommend a recovery action."
)


col1, col2 = st.columns(2)


with col1:

    amount = st.number_input(
        "Transaction Amount",
        min_value=1.0,
        value=4200.0
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "credit_card",
            "debit_card",
            "upi",
            "net_banking"
        ]
    )

    bank_response_code = st.number_input(
        "Bank Response Code",
        min_value=0,
        value=59
    )

    network_latency_ms = st.number_input(
        "Network Latency (ms)",
        min_value=0,
        value=210
    )


with col2:

    fraud_score = st.slider(
        "Fraud Score",
        min_value=0.0,
        max_value=1.0,
        value=0.82
    )

    prior_failed_attempts = st.number_input(
        "Previous Failures (24h)",
        min_value=0,
        value=0
    )

    customer_tenure = st.number_input(
        "Customer Tenure (days)",
        min_value=0,
        value=800
    )

    recurring = st.selectbox(
        "Recurring Subscription?",
        ["No", "Yes"]
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🚀 ANALYZE PAYMENT",
    type="primary",
    use_container_width=True
):

    transaction = {

        "amount": amount,

        "payment_method": payment_method,

        "bank_response_code": bank_response_code,

        "network_latency_ms": network_latency_ms,

        "fraud_score": fraud_score,

        "prior_failed_attempts_24h": prior_failed_attempts,

        "customer_tenure_days": customer_tenure,

        "account_balance_signal": 0.7,

        "hour_of_day": 14,

        "is_recurring_subscription":
            1 if recurring == "Yes" else 0,
    }


    # Prepare transaction for ML model

    row = pd.DataFrame([transaction])

    row_encoded = pd.get_dummies(
        row,
        columns=["payment_method"],
        prefix="pm"
    )


    for c in feature_cols:

        if c not in row_encoded.columns:

            row_encoded[c] = 0


    X = row_encoded[feature_cols]


    # ML prediction

    prediction = clf.predict(X)[0]

    cause = le.inverse_transform(
        [prediction]
    )[0]


    probabilities = clf.predict_proba(X)[0]

    probability_dict = dict(
        zip(
            le.classes_,
            probabilities
        )
    )


    # Recommendation

    try:

        recommendation = rec_engine.recommend_action(
            cause,
            transaction
        )

        recommended_action = recommendation[
            "recommended_action"
        ]

        expected_probability = recommendation[
            "expected_success_prob"
        ]

        action_scores = recommendation[
            "all_action_scores"
        ]

    except Exception as e:

        recommended_action = "Review / retry based on failure cause"

        expected_probability = 0

        action_scores = {}

        st.warning(
            f"Recommendation engine issue: {e}"
        )


    st.divider()

    st.subheader("🤖 AI Diagnosis")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Failure Cause",
            cause.replace("_", " ").title()
        )


    with result_col2:

        confidence = probability_dict[cause]

        st.metric(
            "AI Confidence",
            f"{confidence * 100:.1f}%"
        )


    st.subheader("🎯 Recommended Recovery Action")


    st.info(
    f"""
    **Why this action?**

    The AI identified the failure as **{cause.replace("_", " ").title()}**
    with **{confidence * 100:.1f}% confidence**.

    Based on the customer's transaction context, the recommended
    action has an expected success probability of
    **{expected_probability * 100:.1f}%**.
    """
)


    if expected_probability:

        st.write(
            f"Expected success probability: "
            f"**{expected_probability * 100:.1f}%**"
        )


    if action_scores:

        st.subheader(
            "Recovery Action Scores"
        )

        action_names = {
    "customer_nudge": "Customer Nudge",
    "retry_24h": "Retry after 24h",
    "retry_2h": "Retry after 2h",
    "retry_72h": "Retry after 72h",
    "retry_immediate": "Retry Immediately",
    "suggest_alt_payment": "Suggest Alternate Payment"
}

        action_df = pd.DataFrame(
            {
        "Action": [
            action_names.get(
                action,
                action.replace("_", " ").title()
            )
            for action in action_scores.keys()
        ],
        "Expected Success": list(
            action_scores.values()
        )
    }
)

        action_df[
            "Expected Success"
        ] *= 100

        st.bar_chart(
            action_df.set_index("Action")
        )


# ============================================================
# BUSINESS IMPACT
# ============================================================

st.divider()

st.header("📊 Business Impact")


impact_col1, impact_col2 = st.columns(2)


with impact_col1:

    st.subheader(
        "Recovery Rate"
    )

    recovery_chart = pd.DataFrame(
        {
            "Recovery Rate (%)": [
                baseline_rate * 100,
                ai_rate * 100
            ]
        },
        index=[
            "Static Retry",
            "AI Recovery"
        ]
    )

    st.bar_chart(
        recovery_chart
    )


with impact_col2:

    st.subheader(
        "Revenue Recovered"
    )

    revenue_chart = pd.DataFrame(
        {
            "Revenue": [
                float(
                    summary[
                        "baseline_recovered_value"
                    ]
                ),
                ai_revenue
            ]
        },
        index=[
            "Static Retry",
            "AI Recovery"
        ]
    )

    st.bar_chart(
        revenue_chart
    )


# ============================================================
# EXISTING CHARTS
# ============================================================

st.divider()

st.header("📈 Model & Transaction Insights")


chart_files = [

    (
        "Failure Cause Distribution",
        "chart_1_failure_distribution.png"
    ),

    (
        "AI Classifier Confusion Matrix",
        "chart_2_confusion_matrix.png"
    ),

    (
        "Recovery Rate: AI vs Baseline",
        "chart_3_recovery_rate.png"
    ),

    (
        "Revenue Recovered",
        "chart_4_revenue_recovered.png"
    ),

    (
        "Feature Importance",
        "chart_5_feature_importance.png"
    ),

    (
        "Recommended Action Mix",
        "chart_6_action_mix.png"
    )
]


for title, filename in chart_files:

    chart_path = CHARTS_DIR / filename

    if chart_path.exists():

        st.subheader(title)

        st.image(
            str(chart_path),
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Payment Recovery — Hackathon Prototype | "
    "Synthetic transaction data for demonstration"
)