# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from asyncio import run as run_async
from io import BytesIO
from json import loads, JSONDecodeError
from mimetypes import add_type, guess_type, types_map
from numpy import array_repr, ndarray
from pathlib import Path, PurePath
from PIL import Image
from rich import print_json
from tempfile import mkstemp
from typer import Argument, Context, Option

from ..muna import Muna
from ..logging import CustomProgress, CustomProgressTask
from ..services import Value
from ..types import Prediction
from .auth import get_access_key

def create_prediction(
    tag: str=Argument(..., help="Predictor tag."),
    quiet: bool=Option(False, "--quiet", help="Suppress verbose logging when creating the prediction."),
    context: Context = 0
):
    run_async(_predict_async(tag, quiet=quiet, context=context))

async def _predict_async(tag: str, quiet: bool, context: Context):
    # Preload
    with CustomProgress(transient=True, disable=quiet):
        muna = Muna(get_access_key())
        with CustomProgressTask(
            loading_text="Preloading predictor...",
            done_text="Preloaded predictor"
        ):
            muna.predictions.create(tag, inputs={ })
        with CustomProgressTask(loading_text="Making prediction..."):
            inputs = { }
            for i in range(0, len(context.args), 2):
                name = context.args[i].lstrip("-").replace("-", "_")
                value = _parse_value(context.args[i+1])
                inputs[name] = value
            prediction = muna.predictions.create(tag, inputs=inputs)
    _log_prediction(prediction)

def _parse_value(data: str) -> Value: # CHECK # Add YAML and audio support
    # Add YAML
    if ".yml" not in types_map or ".yaml" not in types_map: # remove in Python 3.14
        add_type("application/yaml", ".yml")
        add_type("application/yaml", ".yaml")
    # Raw string
    if data.startswith("\\"):
        return data[1:]
    # File
    if data.startswith("@"):
        path = Path(data[1:]).expanduser().resolve()
        mime, _ = guess_type(path, strict=False)
        match mime:
            case "application/json":                    return loads(path.read_text())
            case str() if mime.startswith("image/"):    return Image.open(path)
            case str() if mime.startswith("text/"):     return path.read_text()
            case _:                                     return BytesIO(path.read_bytes())
    # JSON
    try:
        return loads(data)
    except JSONDecodeError:
        return data

def _log_prediction(prediction: Prediction):
    images = [value for value in prediction.results or [] if isinstance(value, Image.Image)]
    prediction.results = [_serialize_value(value) for value in prediction.results] if prediction.results is not None else None
    print_json(data=prediction.model_dump())
    for image in images:
        image.show()

def _serialize_value(value) -> str:
    if isinstance(value, ndarray):
        return array_repr(value)
    if isinstance(value, Image.Image):
        _, path = mkstemp(suffix=".png" if value.mode == "RGBA" else ".jpg")
        value.save(path)
        return path
    if isinstance(value, BytesIO):
        return str(value)
    if isinstance(value, PurePath):
        return str(value)
    return value