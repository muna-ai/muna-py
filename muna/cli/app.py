#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

import typer

from ..logging import TracebackMarkupConsole
from ..version import __version__

from .auth import app as auth_app
from .compile import compile_function
from .deploy import app as deployments_app, deploy_function
from .misc import cli_options
from .predictions import create_prediction
from .predictors import app as predictors_app
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
    name="compile",
    help="Compile a model for deployment.",
    rich_help_panel="Compile ⚡"
)(compile_function)

# Predictions
app.command(
    name="predict",
    help="Invoke a compiled model.",
    context_settings={ "allow_extra_args": True, "ignore_unknown_options": True },
    rich_help_panel="Compile ⚡"
)(create_prediction)
app.add_typer(
    predictors_app,
    name="predictors",
    help="Manage your compiled models.",
    rich_help_panel="Compile ⚡"
)

# Deployments
app.command(
    name="deploy",
    help="Deploy a model on Muna.",
    rich_help_panel="Deploy 🚀"
)(deploy_function)
app.add_typer(
    deployments_app,
    name="deployments",
    help="Manage your deployments.",
    rich_help_panel="Deploy 🚀"
)

# Auth
app.add_typer(
    auth_app,
    name="auth",
    help="Login, logout, and check your authentication status.",
    rich_help_panel="Auth 👤"
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
