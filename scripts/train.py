import glob
import json
import os
from typing import List

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
    found = [c for c in candidates if c in df.columns]
    if not found:
        found = [c for c in df.columns if c != "trip_distance" and pd.api.types.is_numeric_dtype(df[c])]
    return found


def train_model():
    """Train a simple regression model on the latest processed snapshot and log to MLflow."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parquet_files = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
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
    if parquet_files:
        try:
            latest_file = max(parquet_files, key=os.path.getctime)
            df = pd.read_parquet(latest_file)
            data_source = latest_file
        except Exception as exc:
            print(f"WARNING: Failed to read parquet files, using synthetic data. Details: {exc}")

    feature_cols = _select_features(df)
    if not feature_cols:
        print("WARNING: No feature columns found; falling back to synthetic defaults.")
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

    target_col = "trip_distance"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found.")

    feature_cols = [c for c in feature_cols if c != target_col]

    numeric_cols = feature_cols + [target_col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 10:
        print(f"WARNING: Not enough clean rows ({len(clean_df)}). Falling back to synthetic data.")
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
        clean_df = df[feature_cols + [target_col]]

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    rmse = mae = r2 = None
    try:
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
                    "data_source": data_source,
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
    except Exception as exc:
        print(f"WARNING: MLflow logging failed: {exc}")
        if rmse is None:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

    metrics_out = os.getenv("METRICS_OUT", "metrics/current.json")
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f)
