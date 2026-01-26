# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel
from rich import print as print_rich
from rich.panel import Panel
from typer import Argument
from typing import Annotated

from ..muna import Muna
from .auth import get_access_key

def triage_predictor(
    reference_code: Annotated[
        str,
        Argument(help="Predictor compilation reference code.")
    ]
):
    muna = Muna(get_access_key())
    error = muna.client.request(
        method="GET",
        path=f"/predictors/triage?referenceCode={reference_code}",
        response_type=_TriagedCompileError
    )
    user_panel = Panel(
        error.user,
        title="User Error",
        title_align="left",
        highlight=True,
        border_style="bright_red"
    )
    internal_panel = Panel(
        error.internal,
        title="Internal Error",
        title_align="left",
        highlight=True,
        border_style="gold1"
    )
    print_rich(user_panel)
    print_rich(internal_panel)

class _TriagedCompileError(BaseModel):
    user: str
    internal: str