import glob
import os
from typing import Optional

import pandas as pd

RAW_DIR = os.getenv("RAW_DIR", "./data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")


def transform_data(raw_path: Optional[str] = None):
    """Feature engineering on the latest raw snapshot."""
    if raw_path is None:
        candidates = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
        if not candidates:
            raise FileNotFoundError("No raw parquet files found for transform.")
        raw_path = max(candidates, key=os.path.getctime)

    df = pd.read_parquet(raw_path)

    # Ensure numeric types for distance before lag/rolling
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")

    # Convert datetime columns
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Create time features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek

    # Sort by pickup time before lag/rolling
    df = df.sort_values("tpep_pickup_datetime")
    df["lag_trip_distance"] = df["trip_distance"].shift(1)
    df["rolling_mean_distance"] = df["trip_distance"].rolling(5).mean()

    df[["lag_trip_distance", "rolling_mean_distance"]] = df[
        ["lag_trip_distance", "rolling_mean_distance"]
    ].fillna(0)
    # Replace any remaining NaNs in distance with 0 for parquet schema consistency
    df["trip_distance"] = df["trip_distance"].fillna(0)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_file = os.path.join(
        PROCESSED_DIR, f"nyc_taxi_processed_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    df.to_parquet(processed_file, index=False)

    print(f"Processed data saved to {processed_file}")
    return processed_file


if __name__ == "__main__":
    transform_data()
