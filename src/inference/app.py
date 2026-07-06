from fastapi import FastAPI

from src.inference.model_loader import load_model
from src.inference.schemas import PredictionRequest, PredictionResponse

import time

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from src.inference.metrics import (
    prediction_latency_seconds,
    prediction_requests_total,
)

app = FastAPI(
    title="Financial Complaints Classifier API",
    description="Inference API for Financial Complaints Classification model loaded from MLflow Model Registry.",
    version="1.0.0",
)

model, model_uri = load_model()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "financial-complaints-classifier",
        "model_uri": model_uri,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_uri": model_uri,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()

    prediction = model.predict([request.complaint_text])[0]

    latency = time.time() - start_time
    prediction_requests_total.inc()
    prediction_latency_seconds.observe(latency)

    return PredictionResponse(
        predicted_category=prediction
    )

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )