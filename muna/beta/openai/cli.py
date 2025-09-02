# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import TypeAdapter
from rich import print as print_rich
from rich.prompt import Prompt
from typer import Argument, Option, Typer
from typing_extensions import Annotated

from ...muna import Muna
from ...cli.auth import get_access_key
from .types import Message

app = Typer(no_args_is_help=True)

@app.command(name="compile", help="Create an LLM predictor.", rich_help_panel="Create")
def compile(
    path: Annotated[str, Argument(help="LLM model path or URI.")],
    tag: Annotated[str, Option(help="Predictor tag.")]
):
    pass

@app.command(name="chat", help="Start a chat session.", rich_help_panel="Run")
def chat(
    tag: Annotated[str, Argument(help="LLM predictor tag.")]
):
    # Preload predictor
    muna = Muna(get_access_key())
    muna.predictions.create(tag=tag, inputs={ })
    # Start loop
    adapter = TypeAdapter(list[Message])
    history = list[Message]()
    while True:
        prompt = Prompt.ask("[dodger_blue2](user)[/dodger_blue2]")
        user_message = Message(role="user", content=prompt)
        history.append(user_message)
        response = ""
        print_rich("[hot_pink](assistant)[/hot_pink]: ", end="")
        for prediction in muna.predictions.stream(
            tag=tag,
            inputs={ "messages": adapter.dump_python(history) }
        ):
            delta_message = prediction.results[0]["choices"][0]["delta"]
            if delta_message:
                delta_content = delta_message["content"]
                response += delta_content
                print_rich(delta_content, end="")
            else:
                print_rich()

@app.command(name="serve", help="Start an LLM server.", rich_help_panel="Run")
def serve(
    port: Annotated[int, Option(help="Port to start the server on.")] = 11435
):
    pass