# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from ..remote.remote import RemotePredictionService
from .speech import SpeechService
from .transcription import TranscriptionService

class AudioService:
    """
    Audio service.
    """
    speech: SpeechService
    transcriptions: TranscriptionService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.speech = SpeechService(predictors, predictions, remote_predictions)
        self.transcriptions = TranscriptionService(predictors, predictions, remote_predictions)