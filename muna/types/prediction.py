#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, Field
from typing import Literal

Acceleration = Literal["local_auto", "local_cpu", "local_gpu", "local_npu"]

class PredictionResource(BaseModel):
    """
    Prediction resource.

    Members:
        type (str): Resource type.
        url (str): Resource URL.
        name (str): Resource name.
    """
    type: str = Field(description="Resource type.")
    url: str = Field(description="Resource URL.")
    name: str | None = Field(default=None, description="Resource name.")

class Prediction(BaseModel):
    """
    Prediction.

    Members:
        id (str): Prediction identifier.
        tag (str): Predictor tag.
        configuration (str): Prediction configuration token. This is only populated for `EDGE` predictions.
        resources (list): Prediction resources. This is only populated for `EDGE` predictions.
        results (list): Prediction results.
        latency (float): Prediction latency in milliseconds.
        error (str): Prediction error. This is `None` if the prediction completed successfully.
        logs (str): Prediction logs.
        created (str): Date created.
    """
    id: str = Field(description="Prediction identifier.")
    tag: str = Field(description="Predictor tag.")
    configuration: str | None = Field(default=None, description="Prediction configuration token. This is only populated for `EDGE` predictions.")
    resources: list[PredictionResource] | None = Field(default=None, description="Prediction resources. This is only populated for `EDGE` predictions.")
    results: list[object] | None = Field(default=None, description="Prediction results.")
    latency: float | None = Field(default=None, description="Prediction latency in milliseconds.")
    error: str | None = Field(default=None, description="Prediction error. This is `None` if the prediction completed successfully.")
    logs: str | None = Field(default=None, description="Prediction logs.")
    created: str = Field(description="Date created.")