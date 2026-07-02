from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    complaint_text: str = Field(
        ...,
        min_length=10,
        description="Customer complaint text to classify.",
    )


class PredictionResponse(BaseModel):
    predicted_category: str