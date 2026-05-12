# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable, Iterator
from contextlib import nullcontext, AbstractContextManager
from numpy import ndarray
from typing import Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..annotations import get_parameter
from .schema import Message

MessageDelegate = Callable[..., Message]

class MessageService: # INCOMPLETE
    """
    Message service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache = dict[str, MessageDelegate]()

    def create(
        self
    ) -> Message:
        """

        """
        pass

    def stream(
        self
    ) -> AbstractContextManager[Iterator[Message]]:
        """

        """
        pass

    def __create_delegate(self, tag: str) -> MessageDelegate:
        pass