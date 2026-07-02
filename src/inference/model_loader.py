import os

import mlflow


DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_MODEL_NAME = "financial_complaints_classifier"
DEFAULT_MODEL_VERSION = "2"


def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_version = os.getenv("MLFLOW_MODEL_VERSION", DEFAULT_MODEL_VERSION)

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    return model, model_uri