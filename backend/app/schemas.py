
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    inference_time_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    device: str
