# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, Field
from requests import put
from rich import print
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from ..muna import Muna
from .auth import get_access_key

app = Typer(no_args_is_help=True)

@app.command(name="upload", help="Upload a prediction resource.")
def upload(
    path: Annotated[Path, Argument(..., help="Path to resource file.", resolve_path=True, exists=True)],
    public: Annotated[bool, Option(..., "--public", help="Whether to make the resource publicly available.")]=False
):
    if not path.is_file():
        raise ValueError(f"Cannot upload resource at path {path} because it is not a file")
    muna = Muna(get_access_key())
    if public:
        _upload_value(path, muna=muna)
    else:
        muna.client.upload(path, progress=True)
        print(f"Uploaded resource [bright_cyan]{hash}[/bright_cyan]")

@app.command(name="download", help="Download a prediction resource.")
def download(
    hash: Annotated[str, Argument(..., help="Prediction resource checksum.")],
    path: Annotated[Path, Option(help="Output path.")]=None
):
    muna = Muna(get_access_key())
    url = f"{muna.client.api_url}/resources/{hash}"
    path = path or Path(hash)
    muna.client.download(url, path)

def _upload_value(
    path: Path,
    *,
    muna: Muna
):
    value = muna.client.request(
        method="POST",
        path="/values",
        body={ "name": path.name },
        response_type=_CreateValueResponse
    )
    with path.open("rb") as f:
        put(value.upload_url, data=f).raise_for_status()
    print(f"Uploaded resource [bright_cyan]{value.download_url}[/bright_cyan]")

class _CreateValueResponse(BaseModel):
    upload_url: str = Field(validation_alias="uploadUrl")
    download_url: str = Field(validation_alias="downloadUrl")