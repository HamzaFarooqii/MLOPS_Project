import glob
import os

import pandas as pd

RAW_DIR = os.getenv("RAW_DIR", "./data/raw")


def run_quality_check():
    """Validate required columns and null ratio on the freshest raw snapshot."""
    latest_file = max(glob.glob(os.path.join(RAW_DIR, "*.parquet")), key=os.path.getctime)
    df = pd.read_parquet(latest_file)

    if df.empty:
        raise ValueError("Data quality failed: raw dataset is empty")

    required_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    null_percent = df[required_columns].isnull().mean()
    if any(null_percent > 0.01):  # >1% nulls
        raise ValueError(f"Data quality check failed, nulls exceed 1%: {null_percent}")

    print(f"Data quality passed for {latest_file}")
    return latest_file
