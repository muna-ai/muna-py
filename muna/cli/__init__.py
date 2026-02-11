# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import typer

from ..logging import TracebackMarkupConsole
from ..version import __version__

from .auth import app as auth_app
from .compile import compile_function, transpile_function
from .misc import cli_options
from .predictions import create_prediction
from .predictors import archive_predictor, delete_predictor, retrieve_predictor
from .resources import app as resources_app
from .triage import triage_predictor

# Define CLI
typer.main.console_stderr = TracebackMarkupConsole()
app = typer.Typer(
    name=f"Muna CLI {__version__}",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
    add_completion=False
)

# Add top level options
app.callback()(cli_options)

# Compilation
app.command(
    name="transpile",
    help="Transpile a Python function to C++ source code.",
    rich_help_panel="Compilation"
)(transpile_function)
app.command(
    name="compile",
    help="Compile a Python function for deployment.",
    rich_help_panel="Compilation"
)(compile_function)

# Functions
app.command(
    name="predict",
    help="Invoke a compiled Python function.",
    context_settings={ "allow_extra_args": True, "ignore_unknown_options": True },
    rich_help_panel="Functions"
)(create_prediction)
app.command(
    name="retrieve",
    help="Retrieve a compiled function.",
    rich_help_panel="Functions"
)(retrieve_predictor)
app.command(
    name="archive",
    help="Archive a compiled function." ,
    rich_help_panel="Functions"
)(archive_predictor)
app.command(
    name="delete",
    help="Delete a compiled function.",
    rich_help_panel="Functions"
)(delete_predictor)

# Subcommands
app.add_typer(
    auth_app,
    name="auth",
    help="Login, logout, and check your authentication status.",
    rich_help_panel="Auth"
)

# Insiders
app.command(
    name="triage",
    help="Triage a compile error.",
    rich_help_panel="Insiders",
    hidden=True
)(triage_predictor)
app.add_typer(
    resources_app,
    name="resources",
    help="Manage prediction resources.",
    rich_help_panel="Insiders",
    hidden=True
)

# Run
if __name__ == "__main__":
    app()