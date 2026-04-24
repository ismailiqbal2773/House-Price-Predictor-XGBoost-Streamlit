import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

TARGET_COLUMN = "Amount(in rupees)"
RAW_COLUMNS = [
    "Title",
    "Amount(in rupees)",
    "Price (in rupees)",
    "location",
    "Carpet Area",
    "Super Area",
    "Floor",
    "Bathroom",
    "Balcony",
    "Car Parking",
    "Transaction",
    "Furnishing",
    "facing",
    "overlooking",
    "Ownership",
]
NUMERIC_FEATURES = [
    "price_per_sqft",
    "carpet_area_sqft",
    "super_area_sqft",
    "floor_number",
    "bathroom_count",
    "balcony_count",
    "parking_count",
    "bhk",
]
DISCRETE_DROPDOWN_FEATURES = [
    "floor_number",
    "bathroom_count",
    "balcony_count",
    "parking_count",
]
CATEGORICAL_FEATURES = [
    "location",
    "transaction",
    "furnishing",
    "facing",
    "overlooking",
    "ownership",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_amount_to_inr(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", " ").replace("\xa0", " ").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    if "cr" in text:
        amount *= 1e7
    elif "lac" in text or "lakh" in text:
        amount *= 1e5
    return amount


def parse_area_to_sqft(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", " ").replace("\xa0", " ").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    area = float(match.group(1))
    if "sq yrd" in text or "sqyrd" in text or "yard" in text:
        area *= 9
    return area


def parse_floor_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "ground" in text:
        return 0.0
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def parse_count(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.search(r"(\d+)", text)
    if match:
        return float(match.group(1))
    if "no" in text:
        return 0.0
    return np.nan


def extract_bhk(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    match = re.search(r"(\d+)\s*bhk", text)
    return float(match.group(1)) if match else np.nan


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return text if text else "unknown"


def clean_display_text(value: Any) -> str:
    if pd.isna(value):
        return "Not Specified"
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else "Not Specified"


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    processed = pd.DataFrame(index=df.index)
    processed["target_amount_inr"] = df[TARGET_COLUMN].apply(parse_amount_to_inr)
    processed["price_per_sqft"] = pd.to_numeric(df["Price (in rupees)"], errors="coerce")
    processed["carpet_area_sqft"] = df["Carpet Area"].apply(parse_area_to_sqft)
    processed["super_area_sqft"] = df["Super Area"].apply(parse_area_to_sqft)
    processed["floor_number"] = df["Floor"].apply(parse_floor_number)
    processed["bathroom_count"] = df["Bathroom"].apply(parse_count)
    processed["balcony_count"] = df["Balcony"].apply(parse_count)
    processed["parking_count"] = df["Car Parking"].apply(parse_count)
    processed["bhk"] = df["Title"].apply(extract_bhk)

    processed["location"] = df["location"].apply(normalize_text)
    processed["transaction"] = df["Transaction"].apply(normalize_text)
    processed["furnishing"] = df["Furnishing"].apply(normalize_text)
    processed["facing"] = df["facing"].apply(normalize_text)
    processed["overlooking"] = df["overlooking"].apply(normalize_text)
    processed["ownership"] = df["Ownership"].apply(normalize_text)

    processed["display_location"] = df["location"].apply(clean_display_text)
    processed["display_transaction"] = df["Transaction"].apply(clean_display_text)
    processed["display_furnishing"] = df["Furnishing"].apply(clean_display_text)
    processed["display_facing"] = df["facing"].apply(clean_display_text)
    processed["display_overlooking"] = df["overlooking"].apply(clean_display_text)
    processed["display_ownership"] = df["Ownership"].apply(clean_display_text)

    processed = processed.dropna(subset=["target_amount_inr"]).copy()

    for column in NUMERIC_FEATURES:
        median_value = processed[column].median()
        processed[column] = processed[column].fillna(median_value)

    for column in CATEGORICAL_FEATURES:
        processed[column] = processed[column].fillna("unknown").astype("category")
        display_column = f"display_{column}"
        processed[display_column] = processed[display_column].fillna("Not Specified")

    lower_bound = processed["target_amount_inr"].quantile(0.01)
    upper_bound = processed["target_amount_inr"].quantile(0.995)
    processed = processed[
        (processed["target_amount_inr"] >= lower_bound)
        & (processed["target_amount_inr"] <= upper_bound)
    ].copy()
    return processed


def format_inr(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:,.2f} Lac"
    return f"₹{amount:,.0f}"


def train_model(processed_df: pd.DataFrame, random_state: int = 42) -> dict[str, Any]:
    sampled_df = processed_df
    if len(processed_df) > 90000:
        sampled_df = processed_df.sample(n=90000, random_state=random_state)

    X = sampled_df[FEATURE_COLUMNS].copy()
    y = sampled_df["target_amount_inr"].copy()

    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=220,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.2,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, np.log1p(y_train))
    preds = np.expm1(model.predict(X_test))
    preds = np.clip(preds, 0, None)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "median_ae": float(np.median(np.abs(y_test.values - preds))),
        "r2": float(r2_score(y_test, preds)),
        "mape": float(
            np.mean(np.abs(y_test.values - preds) / np.maximum(y_test.values, 1)) * 100
        ),
        "rows": int(len(sampled_df)),
        "rows_total": int(len(processed_df)),
    }
    feature_importance = (
        pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
        .astype(float)
    )

    numeric_defaults = {
        column: float(processed_df[column].median()) for column in NUMERIC_FEATURES
    }
    numeric_ranges: dict[str, tuple[float, float]] = {}
    for column in NUMERIC_FEATURES:
        lower = float(processed_df[column].quantile(0.01))
        upper = float(processed_df[column].quantile(0.99))
        if lower == upper:
            upper = lower + 1.0
        numeric_ranges[column] = (lower, upper)

    discrete_numeric_options: dict[str, list[int]] = {}
    for column in DISCRETE_DROPDOWN_FEATURES:
        series = processed_df[column].dropna()
        max_allowed = 80 if column == "floor_number" else 20
        values = sorted(
            {
                int(value)
                for value in series.tolist()
                if float(value).is_integer() and 0 <= value <= max_allowed
            }
        )
        if not values:
            median_val = int(round(numeric_defaults[column]))
            values = [max(median_val, 0)]
        discrete_numeric_options[column] = values

    categorical_options = {
        column: sorted(processed_df[column].astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    categorical_option_pairs: dict[str, list[tuple[str, str]]] = {}
    for column in CATEGORICAL_FEATURES:
        display_column = f"display_{column}"
        mapping_df = (
            processed_df[[column, display_column]]
            .dropna(subset=[column])
            .drop_duplicates()
            .copy()
        )
        canonical_to_display: dict[str, str] = {}
        for canonical_value, group in mapping_df.groupby(column):
            mode_series = group[display_column].mode()
            display_value = (
                mode_series.iloc[0]
                if not mode_series.empty
                else group[display_column].iloc[0]
            )
            canonical_to_display[str(canonical_value)] = str(display_value)

        pairs: list[tuple[str, str]] = []
        for canonical_value in categorical_options[column]:
            display_value = canonical_to_display.get(canonical_value, canonical_value)
            if canonical_value == "unknown":
                display_value = "Not Specified"
            pairs.append((display_value, canonical_value))
        pairs = sorted(pairs, key=lambda pair: pair[0].lower())
        categorical_option_pairs[column] = pairs

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "numeric_defaults": numeric_defaults,
        "numeric_ranges": numeric_ranges,
        "discrete_numeric_options": discrete_numeric_options,
        "categorical_options": categorical_options,
        "categorical_option_pairs": categorical_option_pairs,
    }


def predict_amount(model: XGBRegressor, input_row: dict[str, Any]) -> float:
    frame = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("category")
    prediction = float(np.expm1(model.predict(frame)[0]))
    return max(prediction, 0.0)
