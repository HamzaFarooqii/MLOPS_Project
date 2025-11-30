import glob
import os

import mlflow
import pandas as pd

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def profile_and_log():
    """Generate profiling report for the latest processed file and log to MLflow."""
    # Import heavy dependency inside the task to keep DAG import fast
    from ydata_profiling import ProfileReport

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    latest_file = max(glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")), key=os.path.getctime)
    df = pd.read_parquet(latest_file)

    profile = ProfileReport(df, title="NYC Taxi Data Profiling Report", explorative=True)
    report_file = latest_file.replace(".parquet", "_profile.html")
    profile.to_file(report_file)

    with mlflow.start_run(run_name="profiling"):
        mlflow.log_artifact(report_file, artifact_path="profiles")
        mlflow.log_param("rows", df.shape[0])
        mlflow.log_param("columns", df.shape[1])

    print(f"Pandas profiling report logged to MLflow: {report_file}")
