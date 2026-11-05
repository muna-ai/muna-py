# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from rich import print_json
from typer import Argument, Typer
from typing import Annotated

from ..muna import Muna
from ..logging import CustomProgress, CustomProgressTask
from .auth import get_access_key

app = Typer(no_args_is_help=True)

@app.command(name="retrieve")
def retrieve_predictor(
    tag: Annotated[
        str,
        Argument(help="Predictor tag.")
    ]
):
    """
    Retrieve a compiled model.
    """
    with CustomProgress(transient=True):
        with CustomProgressTask("Retrieving predictor..."):
            muna = Muna(get_access_key())
            predictor = muna.predictors.retrieve(tag)
            predictor = predictor.model_dump() if predictor else None
            print_json(data=predictor)

@app.command(name="archive")
def archive_predictor(
    tag: Annotated[
        str,
        Argument(help="Predictor tag.")
    ]
):
    """
    Archive a compiled model.
    """
    with CustomProgress():
        with CustomProgressTask(
            "Archiving predictor...",
            done_text=f"Archived predictor: [bold dark_orange]{tag}[/bold dark_orange]"
        ):
            muna = Muna(get_access_key())
            muna.client.request(
                method="POST",
                path=f"/predictors/{tag}/archive"
            )

@app.command(name="delete")
def delete_predictor(
    tag: Annotated[
        str,
        Argument(help="Predictor tag.")
    ]
):
    """
    Delete a compiled model.
    """
    with CustomProgress():
        with CustomProgressTask(
            "Deleting predictor...",
            done_text=f"Deleted predictor: [bold red]{tag}[/bold red]"
        ):
            muna = Muna(get_access_key())
            muna.client.request(
                method="DELETE",
                path=f"/predictors/{tag}"
            )
