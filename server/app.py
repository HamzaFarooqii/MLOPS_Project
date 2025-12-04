import os
import time
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/HamzaFarooqii/my-first-repo.mlflow")
MODEL_URI = os.getenv("MODEL_URI", "models:/taxi_rps_model/Production")
MODEL_LOAD_RETRIES = int(os.getenv("MODEL_LOAD_RETRIES", "3"))
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "false").lower() == "true"

app = FastAPI(title="Taxi RPS Model API", version="1.0.0")
mlflow.set_tracking_uri(MLFLOW_URI)


class TripFeatures(BaseModel):
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_dayofweek: int = Field(..., ge=0, le=6)
    lag_trip_distance: float = Field(..., ge=0)
    rolling_mean_distance: float = Field(..., ge=0)
    passenger_count: float = Field(..., ge=0)
    fare_amount: float = Field(..., ge=0)
    trip_distance: float | None = Field(None, ge=0)


def load_model_with_retry() -> Any:
    last_exc: Exception | None = None
    for attempt in range(MODEL_LOAD_RETRIES):
        try:
            return mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to load model after {MODEL_LOAD_RETRIES} attempts: {last_exc}")


# Global model cache
model = None

# Prometheus metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
LATENCY = Histogram("inference_latency_seconds", "Inference latency seconds")
OOD_COUNT = Counter("inference_ood_total", "Requests flagged as out-of-distribution")
DRIFT_RATIO = Gauge("inference_ood_ratio", "Ratio of OOD-like requests")
_drift_totals = {"total": 0, "ood": 0}


@app.on_event("startup")
def _load_model() -> None:
    global model
    if SKIP_MODEL_LOAD:
        class _Stub:
            def predict(self, df):
                return [0.0] * len(df)

        model = _Stub()
        return
    model = load_model_with_retry()


@app.get("/health")
def health() -> Dict[str, str]:
    status = "ok" if model is not None else "loading"
    return {"status": status}


@app.post("/predict")
def predict(payload: TripFeatures) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    REQUEST_COUNT.inc()
    _drift_totals["total"] += 1
    start = time.time()

    df = pd.DataFrame([payload.dict()])

    ood = False
    if "trip_distance" in df and (df["trip_distance"] > 100).any():
        ood = True
    if ood:
        OOD_COUNT.inc()
        _drift_totals["ood"] += 1
    DRIFT_RATIO.set(_drift_totals["ood"] / max(_drift_totals["total"], 1))

    try:
        preds = model.predict(df)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model predict failed: {exc}") from exc
    finally:
        LATENCY.observe(time.time() - start)

    return {"prediction": preds.tolist(), "ood_flag": ood}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
