# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from contextlib import nullcontext
from hashlib import file_digest
from io import SEEK_END, SEEK_SET
from math import ceil
from pathlib import Path
from pydantic import BaseModel, Field
from requests import get, put
from rich.progress import (
    Progress, BarColumn, DownloadColumn, TextColumn,
    TimeRemainingColumn, TransferSpeedColumn
)
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from urllib.parse import urlparse
from .client import MunaAPIError, MunaClient

RESOURCE_URL_BASE = "https://cdn.fxn.ai/resources"
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100 MB
MULTIPART_CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB

def download_resource(
    url: str,
    path: Path,
    *,
    client: MunaClient=None,
    progress: str | bool=True
) -> Path:
    """
    Download a resource to a given path.
    """
    response = get(
        url,
        headers={ "Authorization": f"Bearer {client.access_key}" } if client else None,
        stream=True,
        allow_redirects=True
    )
    response.raise_for_status()
    size = int(response.headers.get("content-length", 0))
    name = Path(urlparse(url).path).name
    completed = 0
    color = progress if isinstance(progress, str) else "dark_orange"
    with (
        Progress(
            TextColumn(f"[{color}]{{task.description}}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            disable=not color
        ) as progress_bar,
        NamedTemporaryFile(mode="wb", delete=False) as tmp_file
    ):
        task_id = progress_bar.add_task(name, total=size)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
                completed += len(chunk)
                progress_bar.update(task_id, total=size, completed=completed)
    Path(tmp_file.name).replace(path)
    return path

def upload_resource(
    path: str | Path | BinaryIO,
    *,
    client: MunaClient,
    progress: bool=True
) -> str:
    """
    Upload a resource and return the resource URL.
    """
    # Handle path or file-like object
    path = Path(path) if isinstance(path, str) else path
    if isinstance(path, Path):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Cannot upload resource at path {path} because it is not a file")
        file_size = path.stat().st_size
        with path.open("rb") as f:
            resource_hash = file_digest(f, "sha256").hexdigest()
    else:
        # Get file size
        current_pos = path.tell()
        path.seek(0, SEEK_END)
        file_size = path.tell()
        path.seek(current_pos, SEEK_SET)
        # Compute hash
        resource_hash = file_digest(path, "sha256").hexdigest()
        path.seek(current_pos, SEEK_SET)
    # Check if resource already exists
    try:
        client.request(method="HEAD", path=f"/resources/{resource_hash}")
        return f"{RESOURCE_URL_BASE}/{resource_hash}"  # Resource already exists
    except MunaAPIError as e:
        if e.status_code != 404:
            raise
    # Upload
    if file_size >= MULTIPART_THRESHOLD:
        _upload_resource_multipart(
            path,
            file_size=file_size,
            resource_hash=resource_hash,
            client=client,
            progress=progress
        )
    else:
        _upload_resource_single(
            path,
            file_size=file_size,
            resource_hash=resource_hash,
            client=client,
            progress=progress
        )
    # Return
    return f"{RESOURCE_URL_BASE}/{resource_hash}"

def _upload_resource_single(
    source: Path | BinaryIO,
    *,
    file_size: int,
    resource_hash: str,
    client: MunaClient,
    progress: bool
) -> str:
    """
    Upload a resource using single upload.
    """
    resource = client.request(
        method="POST",
        path=f"/resources/{resource_hash}",
        response_type=_CreateResourceResponse
    )
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        disable=not progress
    ) as progress_bar:
        task_id = progress_bar.add_task(resource_hash, total=file_size)
        with (source.open("rb") if isinstance(source, Path) else nullcontext(source)) as f:
            reader = _ProgressReader(f.read(), progress_bar, task_id)
            response = put(resource.url, data=reader)
            response.raise_for_status()
            return response.headers.get("ETag", "")

def _upload_resource_multipart(
    source: Path | BinaryIO,
    *,
    file_size: int,
    resource_hash: str,
    client: MunaClient,
    progress: bool
) -> None:
    """
    Upload a resource using multipart upload.
    """
    num_parts = ceil(file_size / MULTIPART_CHUNK_SIZE)
    resource = client.request(
        method="POST",
        path=f"/resources/{resource_hash}/multipart",
        body={ "parts": num_parts },
        response_type=_CreateResourceMultipartResponse
    )
    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            disable=not progress
        ) as progress_bar:
            task_id = progress_bar.add_task(resource_hash, total=file_size)
            etags = list[str]()
            with (source.open("rb") if isinstance(source, Path) else nullcontext(source)) as f:
                for url in resource.urls:
                    etag = _upload_resource_part(
                        f,
                        url=url,
                        progress=progress_bar,
                        task_id=task_id
                    )
                    etags.append(etag)
        parts = [{ "partNumber": i + 1, "etag": etag } for i, etag in enumerate(etags)]
        client.request(
            method="POST",
            path=f"/resources/{resource_hash}/multipart/{resource.upload_id}",
            body={ "parts": parts }
        )
    except Exception as e:
        try:
            client.request(
                method="DELETE",
                path=f"/resources/{resource_hash}/multipart/{resource.upload_id}"
            )
        except:
            pass
        raise e

def _upload_resource_part(
    stream: BinaryIO,
    *,
    url: str,
    progress: Progress,
    task_id: int
) -> str:
    """
    Upload a single part and return ETag.
    """
    chunk = stream.read(MULTIPART_CHUNK_SIZE)
    reader = _ProgressReader(chunk, progress, task_id)
    response = put(url, data=reader)
    response.raise_for_status()
    return response.headers.get("ETag", "")

class _CreateResourceResponse(BaseModel):
    url: str

class _CreateResourceMultipartResponse(BaseModel):
    upload_id: str = Field(validation_alias="uploadId")
    urls: list[str]

class _ProgressReader:

    def __init__(self, data: bytes, progress: Progress, task_id: int):
        self._data = data
        self._offset = 0
        self._progress = progress
        self._task_id = task_id

    def read(self, size: int=-1) -> bytes:
        if size == -1:
            chunk = self._data[self._offset:]
            self._offset = len(self._data)
        else:
            chunk = self._data[self._offset:self._offset + size]
            self._offset += len(chunk)
        self._progress.advance(self._task_id, len(chunk))
        return chunk

    def __len__(self):
        return len(self._data)