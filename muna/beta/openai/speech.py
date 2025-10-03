# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService

SpeechDelegate = Callable[..., object]

class SpeechService: # INCOMPLETE
    """
    Speech service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__remote_predictions = remote_predictions
        self.__cache = dict[str, SpeechDelegate]()