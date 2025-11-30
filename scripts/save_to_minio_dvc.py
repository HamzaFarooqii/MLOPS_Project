import glob
import os
import subprocess

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
DVC_ROOT = os.getenv("DVC_ROOT", "/opt/airflow")
DVC_REMOTE = os.getenv("DVC_REMOTE", "minio_s3")


def save_to_minio_and_dvc():
    """Version the latest processed file with DVC and push to the MinIO-backed remote."""
    latest_file = max(glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")), key=os.path.getctime)

    print(f"Adding {latest_file} to DVC (remote={DVC_REMOTE})")
    subprocess.run(["dvc", "add", latest_file], cwd=DVC_ROOT, check=True)
    subprocess.run(["dvc", "push", "-r", DVC_REMOTE], cwd=DVC_ROOT, check=True)

    print(f"Processed file versioned in DVC and pushed to remote: {latest_file}")
