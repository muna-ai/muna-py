#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from json import loads, JSONDecodeError
from pydantic import BaseModel, TypeAdapter
from requests import request
from typing import AsyncGenerator, Literal, Type, TypeVar

T = TypeVar("T", bound=BaseModel)

class MunaClient:
    
    def __init__(self, access_key: str, api_url: str | None) -> None:
        self.access_key = access_key
        self.api_url = api_url or "https://api.muna.ai/v1"

    def request(
        self,
        *,
        method: Literal["GET", "HEAD", "POST", "PATCH", "DELETE"],
        path: str,
        body: dict[str, object]=None,
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
            json=body,
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
        
    async def stream(
        self,
        *,
        method: Literal["GET", "HEAD", "POST", "PATCH", "DELETE"],
        path: str,
        body: dict[str, object]=None,
        response_type: Type[T]=None
    ) -> AsyncGenerator[T, None]:
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
            json=body,
            headers={
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self.access_key}"
            },
            stream=True
        )
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

class MunaAPIError(Exception):

    def __init__(self, message: str, status_code: int):
        super().__init__(message, status_code)
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return f"{self.message} (Status Code: {self.status_code})"
    
class _APIError(BaseModel):
    message: str

class _ErrorResponse(BaseModel):
    errors: list[_APIError]

def _parse_sse_event(event: str, data: str, type: Type[T]=None) -> T:
    result = { "event": event, "data": loads(data) }
    result = TypeAdapter(type).validate_python(result) if type is not None else result
    return result