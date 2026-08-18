#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from hashlib import file_digest
from hf_transfer import download, multipart_upload
from io import SEEK_END, SEEK_SET
from json import loads, JSONDecodeError
from math import ceil
from os import close as os_close
from pathlib import Path
from pydantic import BaseModel, Field, TypeAdapter
from requests import head, request
from shutil import copyfileobj
from rich.progress import (
    Progress, BarColumn, DownloadColumn, TextColumn,
    TimeRemainingColumn, TransferSpeedColumn
)
from tempfile import mkstemp, NamedTemporaryFile
from typing import BinaryIO, Iterator, Literal, Type, TypeVar
from urllib.parse import urlparse

T = TypeVar("T", bound=BaseModel)

RESOURCE_URL_BASE = "https://cdn.fxn.ai/resources"
MULTIPART_CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB
UPLOAD_MAX_PARALLEL = 16                 # parallel connections for multipart upload
DOWNLOAD_MAX_FILES = 16                          # parallel connections for hf_transfer

class MunaAPIError(Exception):

    def __init__(self, message: str, status_code: int):
        super().__init__(message, status_code)
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return f"{self.message} (Status Code: {self.status_code})"

class MunaClient:
    
    def __init__(
        self,
        access_key: str | None,
        api_url: str | None
    ) -> None:
        self.access_key = access_key
        self.api_url = api_url or "https://api.muna.ai/v1"

    def request(
        self,
        *,
        method: Literal["GET", "HEAD", "POST", "PATCH", "DELETE"],
        path: str,
        body: dict[str, object] | BaseModel | None=None,
        response_type: Type[T]=None
    ) -> T:
        """
        Make a request to a REST endpoint.

        Parameters:
            method (str): Request method.
            path (str): Endpoint path.
            body (dict): Request JSON body.
            response_type (Type): Response type.
        """
        response = request(
            method=method,
            url=f"{self.api_url}{path}",
            json=_coerce_body(body),
            headers={ "Authorization": f"Bearer {self.access_key}" }
        )
        data = response.text
        try:
            data = response.json()
        except JSONDecodeError:
            pass
        if response.ok:
            return response_type(**data) if response_type is not None else None
        else:
            error = _ErrorResponse(**data).errors[0].message if isinstance(data, dict) else data
            raise MunaAPIError(error, response.status_code)

    def stream(
        self,
        *,
        method: Literal["GET", "HEAD", "POST", "PATCH", "DELETE"],
        path: str,
        body: dict[str, object] | BaseModel | None=None,
        response_type: Type[T]=None
    ) -> Iterator[T]:
        """
        Make a request to a REST endpoint and consume the response as a server-sent events stream.

        Parameters:
            method (str): Request method.
            path (str): Endpoint path.
            body (dict): Request JSON body.
            response_type (Type): Response type.
        """
        response = request(
            method=method,
            url=f"{self.api_url}{path}",
            json=_coerce_body(body),
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.access_key}"
            },
            stream=True
        )
        if not response.ok:
            try:
                error = _ErrorResponse(**response.json()).errors[0].message
            except JSONDecodeError:
                error = response.text
            raise MunaAPIError(error, response.status_code)
        event = None
        data: str = ""
        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                break
            line: str = line.strip()
            if line:
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    line_data = line[len("data:"):].strip()
                    data = f"{data}\n{line_data}"
                continue
            if event is not None:
                yield _parse_sse_event(event, data, response_type)
            event = None
            data = ""
        if event or data:
            yield _parse_sse_event(event, data, response_type)

    def download(
        self,
        url: str,
        path: Path,
        *,
        progress: str | bool=True
    ) -> Path:
        """
        Download a resource to a given path.
        """
        name = Path(urlparse(url).path).name
        color = progress if isinstance(progress, str) else "dark_orange"
        headers = {
            "Authorization": f"Bearer {self.access_key}",
            "User-Agent": "muna-py"
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        response = head(url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length is None:
            raise ValueError(f"Muna CDN resource has no Content-Length: {url}")
        size = int(content_length)

        fd, tmp_name = mkstemp(
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".part"
        )
        os_close(fd)
        tmp_path: Path | None = Path(tmp_name)
        try:
            with Progress(
                TextColumn(f"[{color}]{{task.description}}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                disable=not color
            ) as progress_bar:
                task_id = progress_bar.add_task(name, total=size)
                download(
                    url=url,
                    filename=str(tmp_path),
                    max_files=DOWNLOAD_MAX_FILES,
                    chunk_size=MULTIPART_CHUNK_SIZE,
                    parallel_failures=3,
                    max_retries=5,
                    headers=headers,
                    callback=lambda increment: progress_bar.advance(
                        task_id,
                        increment
                    )
                )
            tmp_path.replace(path)
            tmp_path = None
            return path
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def upload(
        self,
        path: str | Path | BinaryIO,
        *,
        progress: bool | Callable[[int], None]=True
    ) -> str:
        """
        Upload a resource and return the resource URL. Pass a callable as
        `progress` to receive byte increments instead of showing the built-in
        progress bar.
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
            self.request(method="HEAD", path=f"/resources/{resource_hash}")
            return f"{RESOURCE_URL_BASE}/{resource_hash}"  # Resource already exists
        except MunaAPIError as e:
            if e.status_code != 404:
                raise
        self.__upload_resource_multipart(
            path,
            file_size=file_size,
            resource_hash=resource_hash,
            progress=progress
        )
        # Return
        return f"{RESOURCE_URL_BASE}/{resource_hash}"

    def __upload_resource_multipart(
        self,
        source: Path | BinaryIO,
        *,
        file_size: int,
        resource_hash: str,
        progress: bool | Callable[[int], None]
    ) -> None:
        """
        Upload a resource using multipart upload. Parts are uploaded over
        parallel connections; part order is preserved for the completion call.
        """
        num_parts = max(1, ceil(file_size / MULTIPART_CHUNK_SIZE))
        resource = self.request(
            method="POST",
            path=f"/resources/{resource_hash}/multipart",
            body={ "parts": num_parts },
            response_type=_CreateResourceMultipartResponse
        )
        try:
            tmp_path: Path | None = None
            try:
                if isinstance(source, Path):
                    upload_path = source
                else:
                    # hf_transfer operates on paths so its Rust workers can
                    # independently seek and read each part. Stage file-like
                    # inputs once, then use the same transfer path.
                    position = source.tell()
                    with NamedTemporaryFile(mode="wb", delete=False) as tmp:
                        tmp_path = Path(tmp.name)
                        try:
                            copyfileobj(source, tmp, length=MULTIPART_CHUNK_SIZE)
                        finally:
                            source.seek(position, SEEK_SET)
                    upload_path = tmp_path
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    disable=not progress if isinstance(progress, bool) else True
                ) as progress_bar:
                    task_id = progress_bar.add_task(resource_hash, total=file_size)
                    callback = progress if not isinstance(progress, bool) else (
                        lambda increment: progress_bar.advance(task_id, increment)
                    )
                    headers = multipart_upload(
                        file_path=str(upload_path),
                        parts_urls=resource.urls,
                        chunk_size=MULTIPART_CHUNK_SIZE,
                        max_files=UPLOAD_MAX_PARALLEL,
                        parallel_failures=UPLOAD_MAX_PARALLEL,
                        max_retries=5,
                        callback=callback
                    )
                etags = [
                    header.get("etag", header.get("ETag", ""))
                    for header in headers
                ]
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass
            parts = [{ "partNumber": i + 1, "etag": etag } for i, etag in enumerate(etags)]
            self.request(
                method="POST",
                path=f"/resources/{resource_hash}/multipart/{resource.upload_id}",
                body={ "parts": parts }
            )
        except Exception as e:
            try:
                self.request(
                    method="DELETE",
                    path=f"/resources/{resource_hash}/multipart/{resource.upload_id}"
                )
            except:
                pass
            raise e

def _parse_sse_event(event: str, data: str, type: Type[T]=None) -> T:
    result = { "event": event, "data": loads(data) }
    result = TypeAdapter(type).validate_python(result) if type is not None else result
    return result

def _coerce_body(body: dict[str, object] | BaseModel | None) -> dict[str, object] | None:
    return (
        body.model_dump(mode="json", by_alias=True)
        if isinstance(body, BaseModel)
        else body
    )

class _APIError(BaseModel):
    message: str

class _ErrorResponse(BaseModel):
    errors: list[_APIError]

class _CreateResourceMultipartResponse(BaseModel):
    upload_id: str = Field(validation_alias="uploadId")
    urls: list[str]