# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from .speech import SpeechService
from .transcriptions import TranscriptionService

class AudioService:
    """
    Audio service.
    """
    speech: SpeechService
    transcriptions: TranscriptionService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.speech = SpeechService(predictors, predictions)
        self.transcriptions = TranscriptionService(predictors, predictions)