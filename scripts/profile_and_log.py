import glob
import os
from typing import Optional

import mlflow
import pandas as pd

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def profile_and_log(processed_path: Optional[str] = None):
    """Generate profiling report for the latest processed file and log to MLflow."""
    # Import heavy dependency inside the task to keep DAG import fast
    from ydata_profiling import ProfileReport

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if processed_path is None:
        candidates = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
        if not candidates:
            raise FileNotFoundError("No processed parquet files found to profile.")
        processed_path = max(candidates, key=os.path.getctime)

    df = pd.read_parquet(processed_path)

    profile = ProfileReport(df, title="NYC Taxi Data Profiling Report", explorative=True)
    report_file = processed_path.replace(".parquet", "_profile.html")
    profile.to_file(report_file)

    with mlflow.start_run(run_name="profiling"):
        mlflow.log_artifact(report_file, artifact_path="profiles")
        mlflow.log_param("rows", df.shape[0])
        mlflow.log_param("columns", df.shape[1])

    print(f"Pandas profiling report logged to MLflow: {report_file}")
    return processed_path


if __name__ == "__main__":
    profile_and_log()
