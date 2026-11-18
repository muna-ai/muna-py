#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from textwrap import dedent

from rich.console import Console

from ..beta.adapters import (
    OpenAICompletionsAdapter,
    OpenAIDepthAdapter,
    OpenAIEmbeddingsAdapter,
    OpenAIImageEditAdapter,
    OpenAIImageGenerationAdapter,
    OpenAITranscriptionsAdapter,
)
from ..types import Signature
from ._syntax import code_panel

def deployment_snippet(
    signature: Signature,
    tag: str,
    base_url: str
) -> str | None:
    """
    Choose the first supported interface without guessing predictor input names.
    """
    for adapter_type in (
        OpenAICompletionsAdapter,
        OpenAIDepthAdapter,
        OpenAIEmbeddingsAdapter,
        OpenAIImageEditAdapter,
        OpenAIImageGenerationAdapter,
        OpenAITranscriptionsAdapter,
    ):
        adapter = adapter_type.from_signature(signature)
        if adapter is not None:
            return openai_snippet(adapter.kind, tag, base_url)
    return None

def print_deployment_example(
    console: Console,
    signature: Signature,
    tag: str,
    base_url: str,
) -> None:
    code = deployment_snippet(signature, tag, base_url)
    if code is None:
        return
    console.print("\n[bold]Try your model[/bold]")
    console.print(code_panel("# Install OpenAI client\n$ pip install openai", "bash", "[bold]Shell[/bold]"))
    console.print()
    console.print("[dim]Set MUNA_ACCESS_KEY to your Muna access key.[/dim]\n")
    console.print(code_panel(code, "python", "[bold]Python[/bold]"))

def openai_snippet(
    kind: str,
    model: str,
    base_url: str
) -> str:
    # All snippets use the official OpenAI client. Credentials stay in the
    # environment; the CLI never embeds an actual key in the example.
    setup = '''\
        from openai import OpenAI
        from os import environ

        # 💥 Create your OpenAI client
        openai = OpenAI(
            base_url=BASE_URL,
            api_key=environ["MUNA_ACCESS_KEY"],
        )
    '''
    examples = {
        "openai.chat.completions.create": '''\
            # 🔥 Stream a chat completion
            stream = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello!"}],
                stream=True,
            )

            # 🚀 Print tokens as they arrive
            for chunk in stream:
                if chunk.choices:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)
            print()
        ''',
        "openai.images.edit.depth": '''\
            # 🔥 Estimate depth from an image
            response = openai.images.edit(
                model=MODEL,
                image=Path("image.png").read_bytes(),
                prompt="Estimate depth",
                output_format="npz",
            )

            # 🚀 Save the depth map
            Path("depth.npz").write_bytes(
                b64decode(response.data[0].b64_json)
            )
        ''',
        "openai.embeddings.create": '''\
            # 🔥 Generate embeddings
            response = openai.embeddings.create(
                model=MODEL,
                input=["Hello, world!"],
            )

            # 🚀 Print the embedding
            print(response.data[0].embedding)
        ''',
        "openai.images.edit": '''\
            # 🔥 Edit your image
            response = openai.images.edit(
                model=MODEL,
                image=Path("image.png").read_bytes(),
                prompt="Turn the scene into a watercolor painting",
            )

            # 🚀 Save the edited image
            Path("edited.png").write_bytes(
                b64decode(response.data[0].b64_json)
            )
        ''',
        "openai.images.generate": '''\
            # 🔥 Generate an image
            response = openai.images.generate(
                model=MODEL,
                prompt="A small cabin beside a lake at sunrise",
            )

            # 🚀 Save the image
            Path("image.png").write_bytes(
                b64decode(response.data[0].b64_json)
            )
        ''',
        "openai.audio.transcriptions.create": '''\
            # 🔥 Transcribe your audio
            response = openai.audio.transcriptions.create(
                model=MODEL,
                file=("audio.wav", Path("audio.wav").read_bytes()),
            )

            # 🚀 Print the transcript
            print(response.text)
        ''',
    }
    if kind in ("openai.images.generate", "openai.images.edit", "openai.images.edit.depth"):
        setup = dedent(setup).replace(
            "from openai import OpenAI\nfrom os import environ",
            "from base64 import b64decode\nfrom openai import OpenAI\nfrom os import environ\nfrom pathlib import Path",
        )
    elif kind == "openai.audio.transcriptions.create":
        setup = dedent(setup).replace(
            "from os import environ",
            "from os import environ\nfrom pathlib import Path",
        )
    return (
        dedent(setup).replace("BASE_URL", repr(base_url)).rstrip() +
        "\n\n" +
        dedent(examples[kind]).replace("MODEL", repr(model)).rstrip()
    )
