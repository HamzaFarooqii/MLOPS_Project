import os
from datetime import datetime

import pandas as pd
import requests

RAW_DIR = os.getenv("RAW_DIR", "./data/raw")
API_ENDPOINT = os.getenv("SODA_ENDPOINT", "https://data.cityofnewyork.us/resource/m6nq-qud6.json")
LIMIT = int(os.getenv("SODA_LIMIT", 1000))  # unauthenticated limits may be lower; increase if using app token
SODA_APP_TOKEN = os.getenv("SODA_APP_TOKEN")


def extract_data():
    """Pull the latest NYC taxi trips from the live SODA API and persist raw data."""
    print(f"Fetching latest data from {API_ENDPOINT} (limit={LIMIT})...")
    headers = {"X-App-Token": SODA_APP_TOKEN} if SODA_APP_TOKEN else {}
    response = requests.get(
        API_ENDPOINT,
        params={"$limit": LIMIT, "$order": "tpep_pickup_datetime DESC"},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    data = pd.DataFrame(response.json())

    if data.empty:
        raise ValueError("No data returned from the API; aborting pipeline.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    data["collected_at"] = timestamp

    os.makedirs(RAW_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DIR, f"nyc_taxi_raw_{timestamp}.parquet")
    data.to_parquet(file_path, index=False)

    print(f"Raw data saved to {file_path}")
    return file_path
