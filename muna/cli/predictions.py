# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ast import literal_eval
from asyncio import run as run_async
from io import BytesIO
from json import loads
from numpy import array, load as load_array, ndarray, save as save_array
from pathlib import Path
from PIL import Image
from pydantic import JsonValue
from rich import print_json
import sounddevice as sd
from tempfile import mkstemp
from typer import Argument, Context, Option
from typing import Annotated

from ..c import Value as CValue
from ..muna import Muna
from ..logging import CustomProgress, CustomProgressTask
from ..types import Dtype, Parameter, Prediction, Signature, Value
from .auth import get_access_key

def create_prediction(
    tag: Annotated[str, Argument(help="Predictor tag.")],
    quiet: Annotated[bool, Option(
        "--quiet",
        help="Suppress verbose logging when creating the prediction."
    )]=False,
    context: Context=0
):
    run_async(_predict_async(tag, quiet=quiet, context=context))

async def _predict_async(
    tag: str,
    quiet: bool,
    context: Context
):
    # Load raw arguments
    raw_args = {
        context.args[i].lstrip("-").replace("-", "_"): context.args[i+1]
        for i in range(0, len(context.args), 2)
    }
    # Preload
    with CustomProgress(transient=True, disable=quiet):
        muna = Muna(get_access_key())
        with CustomProgressTask(
            loading_text="Preloading predictor...",
            done_text="Preloaded predictor"
        ):
            predictor = muna.predictors.retrieve(tag)
            muna.predictions.create(tag, inputs={ })
        with CustomProgressTask(loading_text="Making prediction..."):
            input_params = { param.name: param for param in predictor.signature.inputs }
            inputs = {
                name: _parse_value(data, input_params[name])
                for name, data in raw_args.items()
                if name in input_params
            }
            prediction = muna.predictions.create(tag, inputs=inputs)
    # Log prediction
    print_json(data=_serialize_prediction(prediction, predictor.signature).model_dump())
    _show_prediction_results(prediction, predictor.signature)

def _serialize_prediction(
    prediction: Prediction,
    signature: Signature
) -> Prediction:
    return Prediction(
        id=prediction.id,
        tag=prediction.tag,
        configuration=prediction.configuration,
        resources=prediction.resources,
        results=(
            prediction.results and
            list(map(_serialize_value, prediction.results))
        ),
        latency=prediction.latency,
        error=prediction.error,
        logs=prediction.logs,
        created=prediction.created
    )

def _serialize_value(
    value: Value
) -> JsonValue:
    """
    Serialize a prediction output value.
    """
    match value:
        # case ndarray() if value.dtype == "float32" and parameter.denotation == "audio":
        #     _, path = mkstemp(suffix=".wav")
        #     with CValue.from_object(value) as audio_value:
        #         data = audio_value.serialize(f"audio/wav;rate={parameter.sample_rate}")
        #     Path(path).write_bytes(data)
        #     return path
        case ndarray():
            _, path = mkstemp(suffix=".npy")
            save_array(path, value)
            return path
        case Image.Image():
            _, path = mkstemp(suffix=".png" if value.mode == "RGBA" else ".jpg")
            value.save(path)
            return path
        case BytesIO():
            _, path = mkstemp(suffix=".bin")
            Path(path).write_bytes(value.getvalue())
            return path
        case _: return value

def _show_prediction_results(
    prediction: Prediction,
    signature: Signature
):
    """
    Show prediction results.
    """
    for value, parameter in zip(prediction.results or [], signature.outputs):
        match value:
            case Image.Image():
                value.show()
            case ndarray() if parameter.denotation == "audio":
                sd.play(value, samplerate=parameter.sample_rate)
                sd.wait()

def _parse_value(
    data: str,
    parameter: Parameter
) -> Value:
    """
    Parse a prediction input value from a CLI argument.
    """
    match parameter.dtype:
        case Dtype.null:
            return None
        case Dtype.bool:
            match data.strip().lower():
                case "true":    return True
                case "false":   return False
                case _:         raise ValueError(f"Cannot parse boolean input value from: {data}")
        case Dtype.float32 if parameter.denotation == "audio":
            path = _resolve_path(data)
            with CValue.from_bytes(
                path.read_bytes(),
                f"audio/*;rate={parameter.sample_rate}"
            ) as value:
                return value.to_object()
        case dtype if dtype in _NUMERIC_DTYPES:
            path = _resolve_path(data)
            if path.suffix == ".npy":
                return load_array(path)
            tensor = array(literal_eval(data), dtype=dtype)
            return tensor.item() if len(tensor.shape) == 0 else tensor
        case Dtype.string:
            return data
        case Dtype.list | Dtype.dict:
            value_str = (
                _resolve_path(data).read_text()
                if data.startswith("@")
                else data
            )
            return loads(value_str)
        case Dtype.image:
            return Image.open(_resolve_path(data))
        case Dtype.binary:
            return BytesIO(_resolve_path(data).read_bytes())
        case Dtype.array_list:
            return load_array(_resolve_path(data))
        case _:
            raise ValueError(f"Cannot parse input value with unsupported data type: {parameter.dtype}")    

def _resolve_path(data: str) -> Path:
    return Path(data[1:] if data.startswith("@") else data).expanduser().resolve()

_INTEGER_DTYPES = {
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64,
}
_FLOAT_DTYPES = { Dtype.bfloat16, Dtype.float16, Dtype.float32, Dtype.float64 }
_COMPLEX_DTYPES = { Dtype.complex64, Dtype.complex128 }
_NUMERIC_DTYPES = _INTEGER_DTYPES | _FLOAT_DTYPES | _COMPLEX_DTYPES