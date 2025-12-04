import glob
import os
import subprocess
from typing import Optional

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
DVC_ROOT = os.getenv("DVC_ROOT", "/opt/airflow")
DVC_REMOTE = os.getenv("DVC_REMOTE", "minio_s3")


def save_to_minio_and_dvc(processed_path: Optional[str] = None):
    """Version the latest processed file with DVC and push to the MinIO-backed remote."""
    if processed_path is None:
        candidates = glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
        if not candidates:
            raise FileNotFoundError("No processed parquet files found to version with DVC.")
        processed_path = max(candidates, key=os.path.getctime)

    print(f"Adding {processed_path} to DVC (remote={DVC_REMOTE})")
    subprocess.run(["dvc", "add", processed_path], cwd=DVC_ROOT, check=True)
    subprocess.run(["dvc", "push", "-r", DVC_REMOTE], cwd=DVC_ROOT, check=True)

    print(f"Processed file versioned in DVC and pushed to remote: {processed_path}")
    return processed_path


if __name__ == "__main__":
    save_to_minio_and_dvc()
