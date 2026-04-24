import warnings

import pandas as pd
import streamlit as st

from ml_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    format_inr,
    predict_amount,
    preprocess_data,
    train_model,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_training_artifacts() -> tuple[dict, pd.DataFrame]:
    raw_df = pd.read_csv("house_prices.csv", usecols=RAW_COLUMNS, low_memory=False)
    processed_df = preprocess_data(raw_df)
    artifacts = train_model(processed_df)
    return artifacts, processed_df


st.markdown(
    """
<style>
    :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --ink: #15232f;
        --muted: #6a7784;
        --brand: #0f766e;
        --accent: #f97316;
        --line: #e3e9f0;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, #d7f4ef 0, transparent 38%),
            radial-gradient(circle at 100% 5%, #ffe3ce 0, transparent 34%),
            var(--bg);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .hero {
        background: linear-gradient(120deg, #134e4a, #0f766e);
        color: #f8fffd;
        border-radius: 16px;
        padding: 1.2rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .hero h1 {
        margin: 0;
        font-size: 1.55rem;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 0.45rem 0 0;
        color: #d5f3ef;
    }
    .panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    }
    .result {
        background: linear-gradient(120deg, #fff7ed, #ffe9d6);
        border: 1px solid #ffdfc2;
        color: #7c2d12;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 0.8rem;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.86rem;
    }
    h3 {
        color: var(--ink);
    }
    .stMetric {
        background: #ffffffcc;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.5rem 0.6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

try:
    artifacts, processed = get_training_artifacts()
except Exception as error:
    st.error(f"Model load/train failed: {error}")
    st.stop()

metrics = artifacts["metrics"]
numeric_defaults = artifacts["numeric_defaults"]
numeric_ranges = artifacts["numeric_ranges"]
discrete_numeric_options = artifacts["discrete_numeric_options"]
categorical_option_pairs = artifacts["categorical_option_pairs"]
feature_importance = artifacts["feature_importance"]

label_map = {
    "price_per_sqft": "Price (INR per sqft)",
    "carpet_area_sqft": "Carpet Area (sqft)",
    "super_area_sqft": "Super Area (sqft)",
    "floor_number": "Floor Number",
    "bathroom_count": "Bathrooms",
    "balcony_count": "Balconies",
    "parking_count": "Parking Spaces",
    "bhk": "BHK",
    "location": "Location",
    "transaction": "Transaction Type",
    "furnishing": "Furnishing",
    "facing": "Facing",
    "overlooking": "Overlooking",
    "ownership": "Ownership",
}

st.markdown(
    """
<div class="hero">
    <h1>House Price Prediction Dashboard</h1>
    <p>Target feature: <b>Amount(in rupees)</b>. Enter property details and get clean INR predictions instantly.</p>
</div>
""",
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Model MAE", format_inr(metrics["mae"]))
metric_cols[1].metric("Median Error", format_inr(metrics["median_ae"]))
metric_cols[2].metric("R² Score", f"{metrics['r2']:.3f}")
metric_cols[3].metric("Rows Trained", f"{metrics['rows']:,}")
st.caption(f"Total cleaned rows: {metrics['rows_total']:,} | Training used sampled rows for fast dashboard response.")

left_col, right_col = st.columns([1.08, 0.92], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Property Inputs")
    st.caption("All features below are used to predict final amount in INR.")

    prediction_input: dict = {}
    with st.form("prediction_form", clear_on_submit=False):
        num_left, num_right = st.columns(2, gap="medium")
        for index, feature in enumerate(NUMERIC_FEATURES):
            col = num_left if index % 2 == 0 else num_right
            if feature in discrete_numeric_options:
                options = discrete_numeric_options[feature]
                default_value = int(round(numeric_defaults[feature]))
                default_index = options.index(default_value) if default_value in options else 0
                selected_value = col.selectbox(
                    label_map[feature],
                    options=options,
                    index=default_index,
                )
                prediction_input[feature] = float(selected_value)
            else:
                lower, upper = numeric_ranges[feature]
                default_value = numeric_defaults[feature]
                if upper <= lower:
                    upper = lower + max(abs(lower) * 0.01, 1.0)
                default_value = min(max(default_value, lower), upper)
                step = max((upper - lower) / 150, 1.0)
                prediction_input[feature] = col.number_input(
                    label_map[feature],
                    min_value=float(lower),
                    max_value=float(upper),
                    value=float(default_value),
                    step=float(step),
                )

        cat_left, cat_right = st.columns(2, gap="medium")
        for index, feature in enumerate(CATEGORICAL_FEATURES):
            col = cat_left if index % 2 == 0 else cat_right
            pairs = categorical_option_pairs[feature]
            display_options = [display_value for display_value, _ in pairs]
            display_to_canonical = {
                display_value: canonical_value for display_value, canonical_value in pairs
            }
            default_index = next(
                (
                    idx
                    for idx, (display_value, canonical_value) in enumerate(pairs)
                    if canonical_value == "unknown" or display_value == "Not Specified"
                ),
                0,
            )
            selected_display = col.selectbox(
                label_map[feature],
                options=display_options,
                index=default_index,
            )
            prediction_input[feature] = display_to_canonical[selected_display]

        submit = st.form_submit_button("Predict House Price", type="primary", use_container_width=True)

    if submit:
        estimate = predict_amount(artifacts["model"], prediction_input)
        carpet_area = max(prediction_input["carpet_area_sqft"], 1.0)
        rate_estimate = estimate / carpet_area
        st.markdown(
            f"""
            <div class="result">
                <h3 style="margin:0;">Predicted Amount: {format_inr(estimate)}</h3>
                <div style="margin-top:0.35rem;">Approx. rate based on carpet area: <b>₹{rate_estimate:,.0f} / sqft</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Prediction generated from trained model on cleaned historical listings.")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Insights")
    st.markdown(
        f'<div class="small-note">MAPE: <b>{metrics["mape"]:.2f}%</b> | Feature set: <b>{len(FEATURE_COLUMNS)}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Feature Importance**")
    top_features = feature_importance.head(10).sort_values(ascending=True)
    st.bar_chart(top_features)

    st.markdown("**Top Locations by Median Price**")
    top_locations = (
        processed.groupby("display_location")["target_amount_inr"]
        .median()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)
    )
    st.bar_chart(top_locations)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Fast mode enabled: model training is cached, so repeated dashboard use remains lightweight."
)
