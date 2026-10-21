#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import AliasChoices, BaseModel, Field
from typing import Literal

from .value import RemoteValue

Acceleration = Literal[
    "local_auto",
    "local_cpu",
    "local_gpu",
    "local_npu",
    "remote_auto",
    "remote_cpu",
    "remote_a10",
    "remote_l40s",
    "remote_a100",
    "remote_h100",
    "remote_h200",
    "remote_b200",
    "remote_mi350x",
    "remote_mi355x",
    "remote_qaic100"
] | str

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
    name: str | None = Field(None, description="Resource name.")

class PredictionRequirements(BaseModel):
    """
    Hardware requirements for a prediction.

    Members:
        gpu_count (int): Number of GPUs the predictor uses. `None` if it requires no GPU.
        gpu_architecture (str): Required GPU compute architecture (e.g. `sm_100`).
        disk_bytes (int): Total size of the predictor's resources in bytes.
    """
    gpu_count: int | None = Field(
        default=None,
        description="Number of GPUs the predictor uses. `None` if it requires no GPU.",
        serialization_alias="gpuCount",
        validation_alias=AliasChoices("gpu_count", "gpuCount")
    )
    gpu_architecture: str | None = Field(
        default=None,
        description="Required GPU compute architecture (e.g. `sm_100`).",
        serialization_alias="gpuArchitecture",
        validation_alias=AliasChoices("gpu_architecture", "gpuArchitecture")
    )
    disk_bytes: int | None = Field(
        default=None,
        description="Total size of the predictor's resources in bytes.",
        serialization_alias="diskBytes",
        validation_alias=AliasChoices("disk_bytes", "diskBytes")
    )

class Prediction(BaseModel):
    """
    Prediction.

    Members:
        id (str): Prediction identifier.
        tag (str): Predictor tag.
        configuration (str): Prediction configuration token.
        resources (list): Prediction resources.
        requirements (PredictionRequirements): Prediction hardware requirements.
        results (list): Prediction results.
        latency (float): Prediction latency in milliseconds.
        error (str): Prediction error. This is `None` if the prediction completed successfully.
        logs (str): Prediction logs.
        created (str): Date created.
    """
    id: str = Field(description="Prediction identifier.")
    tag: str = Field(description="Predictor tag.")
    configuration: str | None = Field(None, description="Prediction configuration token.")
    resources: list[PredictionResource] | None = Field(None, description="Prediction resources.")
    requirements: PredictionRequirements | None = Field(None, description="Prediction hardware requirements.")
    results: list[object] | None = Field(None, description="Prediction results.")
    latency: float | None = Field(None, description="Prediction latency in milliseconds.")
    error: str | None = Field(None, description="Prediction error. This is `None` if the prediction completed successfully.")
    logs: str | None = Field(None, description="Prediction logs.")
    created: str = Field(description="Date created.")

class RemotePrediction(Prediction):
    """
    Remote prediction.
    """
    results: list[RemoteValue] | None = Field(None, description="Prediction results.")