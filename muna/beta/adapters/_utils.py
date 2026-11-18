#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from datetime import datetime

from ...services.prediction import _parse_remote_value
from ...types import Dtype, Parameter, Prediction, RemoteValue, Signature
from ..errors import prediction_error_to_exception

def get_parameter(
    parameters: list[Parameter],
    *,
    dtype: Dtype | set[Dtype],
    denotation: str | None=None
) -> tuple[int | None, Parameter | None]:
    """
    Get a parameter with the given data type and denotation.
    """
    dtype = dtype if isinstance(dtype, set) else { dtype }
    matches = [
        (idx, param)
        for idx, param in enumerate(parameters)
        if (
            param.dtype in dtype and
            denotation in { None, param.denotation }
        )
    ]
    if len(matches) != 1:
        return None, None
    return matches[0]

def validate_inputs(
    signature: Signature,
    parameters: list[Parameter | None]
) -> bool:
    names = [p.name for p in signature.inputs]
    if len(names) != len(set(names)):
        return False
    supported = {p.name for p in parameters if p is not None}
    return all(p.optional or p.name in supported for p in signature.inputs)

def result(prediction: Prediction, index: int) -> object:
    if prediction.error:
        raise prediction_error_to_exception(prediction.error)
    if prediction.results is None or index >= len(prediction.results):
        raise ValueError("Prediction did not return the adapter's output")
    return prediction.results[index]

def decode(value: object, dtype: Dtype) -> object:
    if isinstance(value, RemoteValue):
        if value.dtype != dtype:
            raise ValueError(f"Expected {dtype} result, received {value.dtype}")
        return _parse_remote_value(value)
    return value

def inline_base64(value: RemoteValue, mime_types: set[str]) -> str | None:
    # A matching data URL can pass through without unpacking or re-encoding it.
    if not isinstance(value.data, str) or not value.data.startswith("data:"):
        return None
    header, separator, payload = value.data.partition(",")
    mime = header[5:].split(";", 1)[0].lower()
    if separator and header.endswith(";base64") and mime in mime_types:
        return payload
    return None

def created(prediction: Prediction) -> int:
    return int(datetime.fromisoformat(prediction.created.replace("Z", "+00:00")).timestamp())

INTEGERS = {
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64
}
FLOATS = {
    Dtype.float16, Dtype.bfloat16, Dtype.float32, Dtype.float64
}