import glob
import os
from typing import List

import json
import mlflow
import pandas as pd

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "taxi_rps_model")


def _select_features(df: pd.DataFrame) -> List[str]:
    """Return feature column names used for training."""
    candidates = [
        "pickup_hour",
        "pickup_dayofweek",
        "lag_trip_distance",
        "rolling_mean_distance",
        "passenger_count",
        "fare_amount",
    ]
    return [c for c in candidates if c in df.columns]


def train_model():
    """Train a simple regression model on the latest processed snapshot and log to MLflow."""
    # Import heavy deps inside the task to avoid slowing DAG parsing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parquet_files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
    if parquet_files:
        latest_file = max(parquet_files, key=os.path.getctime)
        df = pd.read_parquet(latest_file)
        data_source = latest_file
    else:
        # Fallback synthetic data for CI environments with no processed artifacts
        data_source = "synthetic"
        df = pd.DataFrame(
            {
                "pickup_hour": list(range(100)),
                "pickup_dayofweek": [i % 7 for i in range(100)],
                "lag_trip_distance": [1.0 + i * 0.01 for i in range(100)],
                "rolling_mean_distance": [1.2 + i * 0.01 for i in range(100)],
                "passenger_count": [1 for _ in range(100)],
                "fare_amount": [5.0 + i * 0.02 for i in range(100)],
                "trip_distance": [1.5 + i * 0.015 for i in range(100)],
            }
        )

    feature_cols = _select_features(df)
    if not feature_cols:
        raise ValueError("No feature columns available for training.")

    target_col = "trip_distance"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found.")

    # Ensure target is not accidentally part of features
    feature_cols = [c for c in feature_cols if c != target_col]

    # Coerce numeric columns and drop rows with NaNs
    numeric_cols = feature_cols + [target_col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 50:
        raise ValueError(f"Not enough clean rows to train (got {len(clean_df)} after dropping NaNs).")

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="rf_trip_distance"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "test_size": 0.2,
                "random_state": 42,
                "features_used": ",".join(feature_cols),
            }
        )
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        if data_source != "synthetic":
            mlflow.log_artifact(data_source, artifact_path="data_used")
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(
            f"Training complete. RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}. "
            f"Logged to MLflow run {mlflow.active_run().info.run_id}"
        )

    # Persist metrics locally for CI/CML comparison
    metrics_out = os.getenv("METRICS_OUT", "metrics/current.json")
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f)
