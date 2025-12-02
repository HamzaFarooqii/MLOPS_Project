import os
import time
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/HamzaFarooqii/my-first-repo.mlflow")
MODEL_URI = os.getenv("MODEL_URI", "models:/taxi_rps_model/Production")

app = FastAPI(title="Taxi RPS Model API", version="1.0.0")
mlflow.set_tracking_uri(MLFLOW_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
LATENCY = Histogram("inference_latency_seconds", "Inference latency seconds")
DRIFT_RATIO = Counter("drift_ratio", "Count of OOD-like requests")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    REQUEST_COUNT.inc()
    start = time.time()

    # Simple DataFrame build; adjust schema as needed
    df = pd.DataFrame([payload])

    # Naive drift proxy: flag if trip_distance unusually large
    if "trip_distance" in df and (df["trip_distance"] > 100).any():
        DRIFT_RATIO.inc()

    preds = model.predict(df)
    LATENCY.observe(time.time() - start)
    return {"prediction": preds.tolist()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
