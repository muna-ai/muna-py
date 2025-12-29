# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from rich import print_json
from typer import Argument, Option
from typing_extensions import Annotated

from ..muna import Muna
from ..logging import CustomProgress, CustomProgressTask
from .auth import get_access_key

def retrieve_source(
    prediction: Annotated[str, Option(help="Prediction identifier. If specified, this MUST be from a prediction returned by the Muna API.")] = None,
    predictor: Annotated[str, Option(help="Predictor tag. If specified, a prediction will be made with this predictor before retrieving the source.")] = None,
    output: Annotated[Path, Option(help="Path to output source file.")] = Path("predictor.cpp")
):
    if not ((predictor is not None) ^ (prediction is not None)):
        raise ValueError(f"Predictor tag or prediction identifier must be provided, but not both.")
    muna = Muna(get_access_key())
    with CustomProgress(transient=True):
        if prediction is None:
            with CustomProgressTask(loading_text="Creating prediction..."):
                empty_prediction = muna.predictions.create(tag=predictor)
                prediction = empty_prediction.id
        with CustomProgressTask(loading_text="Retrieving source..."):
            source = muna.client.request(
                method="GET",
                path=f"/predictions/{prediction}/source",
                response_type=_PredictionSource
            )
            output.write_text(source.code)
            source.code = str(output.resolve())
            print_json(data=source.model_dump(mode="json", by_alias=True))

class _PredictionSource(BaseModel):
    tag: str
    target: str
    code: str
    created: datetime
    compiled: datetime
    latency: float # millis