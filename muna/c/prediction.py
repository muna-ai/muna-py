#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import byref, c_double, c_int32, c_void_p, create_string_buffer
from typing import final

from .fxnc import get_fxnc, status_to_error, FXNStatus
from .map import ValueMap

@final
class Prediction:

    def __init__(self, prediction):
        self.__prediction = prediction

    @property
    def id(self) -> str:
        id = create_string_buffer(256)
        status = get_fxnc().FXNPredictionGetID(self.__prediction, id, len(id))
        if status == FXNStatus.OK:
            return id.value.decode("utf-8")
        else:
            raise RuntimeError(f"Failed to get prediction id with error: {status_to_error(status)}")
    
    @property
    def latency(self) -> float:
        latency = c_double()
        status = get_fxnc().FXNPredictionGetLatency(self.__prediction, byref(latency))
        if status == FXNStatus.OK:
            return latency.value
        else:
            raise RuntimeError(f"Failed to get prediction latency with error: {status_to_error(status)}")

    @property
    def results(self) -> ValueMap | None:
        map = c_void_p()
        status = get_fxnc().FXNPredictionGetResults(self.__prediction, byref(map))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get prediction results with error: {status_to_error(status)}")
        map = ValueMap(map, owner=False)
        return map if len(map) > 0 else None

    @property
    def error(self) -> str | None:
        error = create_string_buffer(2048)
        get_fxnc().FXNPredictionGetError(self.__prediction, error, len(error))
        error = error.value.decode("utf-8")
        return error if error else None

    @property
    def logs(self) -> str:
        fxnc = get_fxnc()
        log_length = c_int32()
        status = fxnc.FXNPredictionGetLogLength(self.__prediction, byref(log_length))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get prediction log length with error: {status_to_error(status)}")
        logs = create_string_buffer(log_length.value + 1)
        status = fxnc.FXNPredictionGetLogs(self.__prediction, logs, len(logs))
        if status == FXNStatus.OK:
            return logs.value.decode("utf-8")
        else:
            raise RuntimeError(f"Failed to get prediction logs with error: {status_to_error(status)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__prediction:
            get_fxnc().FXNPredictionRelease(self.__prediction)
        self.__prediction = None