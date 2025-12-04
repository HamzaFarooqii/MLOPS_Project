import glob
import os
from typing import Optional

import pandas as pd

RAW_DIR = os.getenv("RAW_DIR", "./data/raw")


def run_quality_check(file_path: Optional[str] = None):
    """Validate required columns and null ratio on the freshest raw snapshot."""
    if file_path is None:
        candidates = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
        if not candidates:
            raise FileNotFoundError("No raw parquet files found for quality check.")
        file_path = max(candidates, key=os.path.getctime)

    df = pd.read_parquet(file_path)

    if df.empty:
        raise ValueError("Data quality failed: raw dataset is empty")

    required_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    null_percent = df[required_columns].isnull().mean()
    if any(null_percent > 0.01):  # >1% nulls
        raise ValueError(f"Data quality check failed, nulls exceed 1%: {null_percent}")

    print(f"Data quality passed for {file_path}")
    return file_path


if __name__ == "__main__":
    run_quality_check()
