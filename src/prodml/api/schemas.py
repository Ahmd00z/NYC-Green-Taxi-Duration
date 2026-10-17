"""Pydantic models = the contract of the API. Bad input is rejected here (422)
before it ever reaches the model."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    PULocationID: int = Field(..., gt=0, description="Pickup location ID")
    DOLocationID: int = Field(..., gt=0, description="Dropoff location ID")
    trip_distance: float = Field(
        ..., gt=0, lt=200, description="Trip distance in miles"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5}
        }
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted trip duration in minutes")
    model_version: str
    correlation_id: str = Field(..., description="Same value as X-Request-ID header")
    latency_ms: float = Field(..., description="Time spent inside the model")


class BatchPredictionRequest(BaseModel):
    inputs: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5},
                    {"PULocationID": 132, "DOLocationID": 230, "trip_distance": 12.8},
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float
