# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from datetime import datetime

def split_tag(tag: str) -> tuple[str, str]:
    """
    Split `@owner/name` into `(owner, name)`. Tags without a slash are
    their own name with an empty owner.
    """
    owner, sep, name = tag.partition("/")
    if not sep:
        return "", tag.lstrip("@")
    return owner.lstrip("@"), name

def rfc3339_to_unix(value: str) -> int | None:
    """
    `YYYY-MM-DDTHH:MM:SS[.fff]Z` (the API's date format) to Unix seconds.
    `None` for anything else, including offsets other than `Z`.
    """
    if not value.endswith("Z") or "T" not in value:
        return None
    try:
        timestamp = int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None
    return timestamp if timestamp >= 0 else None
