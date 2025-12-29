# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from rich import print, print_json
from typer import Argument, Typer

from ..muna import Muna

app = Typer(no_args_is_help=True)

@app.command(name="login", help="Login to Muna.")
def login(
    access_key: str=Argument(..., help="Muna access key.", envvar="MUNA_ACCESS_KEY")
):
    muna = Muna(access_key=access_key)
    user = muna.users.retrieve()
    user = user.model_dump() if user else None
    _set_access_key(access_key if user is not None else None)
    print_json(data=user)

@app.command(name="status", help="Get current authentication status.")
def auth_status():
    muna = Muna(get_access_key())
    user = muna.users.retrieve()
    user = user.model_dump() if user else None
    print_json(data=user)

@app.command(name="logout", help="Logout from Muna.")
def logout():
    _set_access_key(None)
    print("Successfully logged out of Muna")

def get_access_key() -> str:
    """
    Get the CLI access key.

    Returns:
        str: CLI access key.
    """
    credentials_path = Path.home() / ".fxn" / "credentials"
    if not credentials_path.exists():
        return None
    with open(credentials_path) as f:
        return f.read()

def _set_access_key(key: str):
    """
    Set the CLI access key.

    Parameters:
        key (str); CLI access key.
    """
    credentials_path = Path.home() / ".fxn" / "credentials"
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    if key:
        with open(credentials_path, "w") as f:
            f.write(key)
    elif credentials_path.exists():
        credentials_path.unlink()