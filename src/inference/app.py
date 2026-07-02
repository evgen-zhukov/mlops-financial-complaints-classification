from fastapi import FastAPI

from src.inference.model_loader import load_model
from src.inference.schemas import PredictionRequest, PredictionResponse


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
    prediction = model.predict([request.complaint_text])[0]

    return PredictionResponse(
        predicted_category=prediction
    )