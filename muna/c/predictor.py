#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import byref, c_void_p
from typing import final

from .configuration import Configuration
from .fxnc import get_fxnc, status_to_error, FXNStatus
from .map import ValueMap
from .prediction import Prediction
from .stream import PredictionStream

@final
class Predictor:

    def __init__(self, configuration: Configuration):
        predictor = c_void_p()
        status = get_fxnc().FXNPredictorCreate(configuration._Configuration__configuration, byref(predictor))
        if status == FXNStatus.OK:
            self.__predictor = predictor
        else:
            raise RuntimeError(f"Failed to create predictor with error: {status_to_error(status)}")

    def create_prediction(self, inputs: ValueMap) -> Prediction:
        prediction = c_void_p()
        status = get_fxnc().FXNPredictorCreatePrediction(
            self.__predictor,
            inputs._ValueMap__map,
            byref(prediction)
        )
        if status == FXNStatus.OK:
            return Prediction(prediction)
        else:
            raise RuntimeError(f"Failed to create prediction with error: {status_to_error(status)}")

    def stream_prediction(self, inputs: ValueMap) -> PredictionStream:
        stream = c_void_p()
        status = get_fxnc().FXNPredictorStreamPrediction(
            self.__predictor,
            inputs._ValueMap__map,
            byref(stream)
        )
        if status == FXNStatus.OK:
            return PredictionStream(stream)
        else:
            raise RuntimeError(f"Failed to stream prediction with error: {status_to_error(status)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__predictor:
            get_fxnc().FXNPredictorRelease(self.__predictor)
        self.__predictor = None