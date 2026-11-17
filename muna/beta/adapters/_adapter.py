#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import Generic, TypeVar

Source = TypeVar("Source", contravariant=True)
Options = TypeVar("Options", contravariant=True)
Response = TypeVar("Response", covariant=True)

class Adapter(
    BaseModel,
    Generic[Source, Options, Response],
    ABC,
    **ConfigDict(frozen=True, extra="forbid")
):
    """
    Abstract response adapter with validated, frozen configuration.
    """
    kind: str

    @abstractmethod
    def adapt(
        self,
        prediction: Source,
        /,
        options: Options | None = None
    ) -> Response:
        ...
