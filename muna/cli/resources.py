# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, Field
from requests import get, put
from rich import print
from rich.progress import BarColumn, DownloadColumn, TimeRemainingColumn, TransferSpeedColumn
from tempfile import NamedTemporaryFile
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from ..logging import CustomProgress, CustomProgressTask
from ..muna import Muna
from ..resources import upload_resource
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
        upload_resource(path, muna=muna, progress=True)
        print(f"Uploaded resource [bright_cyan]{hash}[/bright_cyan]")

@app.command(name="download", help="Download a prediction resource.")
def download(
    hash: Annotated[str, Argument(..., help="Prediction resource checksum.")],
    path: Annotated[Path, Option(help="Output path.")]=None
):
    muna = Muna(get_access_key())
    with CustomProgress():
        api_request_url = f"{muna.client.api_url}/resources/{hash}"
        response = get(
            api_request_url,
            headers={ "Authorization": f"Bearer {muna.client.access_key}" },
            stream=True,
            allow_redirects=True
        )
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        output_path = path or Path(hash)
        completed = 0
        with (
            CustomProgressTask(
                loading_text=f"[dark_orange]{hash}[/dark_orange]",
                done_text=f"Downloaded: [bold dark_orange]{output_path}[/bold dark_orange]",
                columns=[
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn()
                ]
            ) as task,
            NamedTemporaryFile(mode="wb", delete=False) as tmp_file
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    completed += len(chunk)
                    task.update(total=total_size, completed=completed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Path(tmp_file.name).replace(output_path)

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