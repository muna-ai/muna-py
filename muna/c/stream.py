#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import byref, c_void_p
from typing import final

from .fxnc import get_fxnc, status_to_error, FXNStatus
from .prediction import Prediction

@final
class PredictionStream:

    def __init__(self, stream):
        self.__stream = stream

    def __iter__(self):
        return self
    
    def __next__(self) -> Prediction:
        prediction = c_void_p()
        status = get_fxnc().FXNPredictionStreamReadNext(self.__stream, byref(prediction))
        if status == FXNStatus.ERROR_INVALID_OPERATION:
            raise StopIteration()
        elif status != FXNStatus.OK:
            raise RuntimeError(f"Failed to read next prediction in stream with error: {status_to_error(status)}")
        else:
            return Prediction(prediction)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__stream:
            get_fxnc().FXNPredictionStreamRelease(self.__stream)
        self.__stream = None