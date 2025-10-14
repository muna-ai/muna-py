# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

import typer

from ..logging import TracebackMarkupConsole
from ..version import __version__

from .auth import app as auth_app
from .compile import compile_predictor, triage_predictor
from .misc import cli_options
from .predictions import create_prediction
from .predictors import archive_predictor, delete_predictor, retrieve_predictor
from .resources import app as resources_app
from .sources import retrieve_source

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

# Predictions
app.command(
    name="predict",
    help="Make a prediction.",
    context_settings={ "allow_extra_args": True, "ignore_unknown_options": True },
    rich_help_panel="Predictions"
)(create_prediction)
app.command(
    name="source",
    help="Retrieve the generated C++ code for a given prediction.",
    rich_help_panel="Predictions"
)(retrieve_source)

# Predictors
app.command(
    name="compile",
    help="Create a predictor by compiling a Python function.",
    rich_help_panel="Predictors"
)(compile_predictor)
app.command(
    name="retrieve",
    help="Retrieve a predictor.",
    rich_help_panel="Predictors"
)(retrieve_predictor)
app.command(
    name="archive",
    help="Archive a predictor." ,
    rich_help_panel="Predictors"
)(archive_predictor)
app.command(
    name="delete",
    help="Delete a predictor.",
    rich_help_panel="Predictors"
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