import warnings

import pandas as pd

from ml_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    format_inr,
    predict_amount,
    preprocess_data,
    train_model,
)

warnings.filterwarnings("ignore")

raw_df = pd.read_csv("house_prices.zip", usecols=RAW_COLUMNS, low_memory=False)
print(f"Raw shape: {raw_df.shape}")

processed_df = preprocess_data(raw_df)
print(f"Processed rows: {processed_df.shape[0]:,}")
print(
    f"Target range: {format_inr(processed_df['target_amount_inr'].min())} to "
    f"{format_inr(processed_df['target_amount_inr'].max())}"
)

artifacts = train_model(processed_df)
metrics = artifacts["metrics"]

print("\nModel Evaluation")
print("-" * 40)
print(f"MAE       : {format_inr(metrics['mae'])}")
print(f"Median AE : {format_inr(metrics['median_ae'])}")
print(f"R2        : {metrics['r2']:.3f}")
print(f"MAPE      : {metrics['mape']:.2f}%")
print(f"Rows used : {metrics['rows']:,} (from total {metrics['rows_total']:,})")

feature_rank = artifacts["feature_importance"].head(8)
print("\nTop Features")
for name, score in feature_rank.items():
    print(f"  {name:<18} {score:.4f}")

sample = {}
for name in NUMERIC_FEATURES:
    sample[name] = artifacts["numeric_defaults"][name]
for name in CATEGORICAL_FEATURES:
    options = artifacts["categorical_options"][name]
    sample[name] = options[0] if options else "unknown"

sample_prediction = predict_amount(artifacts["model"], sample)
print("\nSample Prediction")
print(f"  Input (median-like sample) -> {format_pkr(sample_prediction)}")
