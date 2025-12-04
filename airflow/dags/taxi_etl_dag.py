from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys

# Make packaged scripts importable inside the Airflow container
sys.path.insert(0, "/opt/airflow/scripts")

from extract import extract_data
from quality_check import run_quality_check
from transform import transform_data
from profile_and_log import profile_and_log
from save_to_minio_dvc import save_to_minio_and_dvc
from train import train_model


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "taxi_rps_pipeline",
    default_args=default_args,
    description="Real-Time Predictive System for NYC Taxi data with DQC, profiling, and DVC/MinIO persistence",
    schedule_interval="@daily",
    start_date=days_ago(1),  # start immediately
    catchup=False,
    tags=["rps", "nyc-taxi"],
) as dag:

    t1_extract = PythonOperator(
        task_id="extract_live_data",
        python_callable=extract_data,
    )

    t2_quality = PythonOperator(
        task_id="data_quality_check",
        python_callable=run_quality_check,
        op_kwargs={"file_path": "{{ ti.xcom_pull(task_ids='extract_live_data') }}"},
    )

    t3_transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
        op_kwargs={"raw_path": "{{ ti.xcom_pull(task_ids='data_quality_check') }}"},
    )

    t4_profile = PythonOperator(
        task_id="profile_and_log",
        python_callable=profile_and_log,
        op_kwargs={"processed_path": "{{ ti.xcom_pull(task_ids='transform_data') }}"},
    )

    t5_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={"processed_path": "{{ ti.xcom_pull(task_ids='transform_data') }}"},
    )

    t6_save = PythonOperator(
        task_id="save_to_minio_dvc",
        python_callable=save_to_minio_and_dvc,
        op_kwargs={"processed_path": "{{ ti.xcom_pull(task_ids='transform_data') }}"},
    )

    t1_extract >> t2_quality >> t3_transform >> t4_profile >> t5_train >> t6_save
