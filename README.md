# Muna for Python

![muna transpile](media/transpile.gif)

Run AI models anywhere.

## Installing Muna
Muna is distributed on PyPi. This distribution contains both the Python client and the command line interface (CLI). To install, open a terminal and run the following command:
```sh
# Install Muna
$ pip install --upgrade muna
```

> [!NOTE]
> Muna requires Python 3.11+

## Transpiling a Python Function
Muna can transpile Python functions into C++, generating a self-contained header-only library that pulls all of its dependencies automatically (e.g. llama.cpp, mlx, CUDA). First, add the `@compile` decorator to your function:
```py
from muna import compile

@compile()
def do_stuff():
    ...
```

Then use the Muna CLI to transpile to C++:
```sh
# Transpile the Python function to C++
$ muna transpile do_stuff.py
```

Muna will create a cloud sandbox to setup your Python function, trace it, lower to C++, then generate a folder containing the header-only library and a corresponding `CMakeLists.txt`.

> [!TIP]
> Even though the compiler is not open-source, you can [read up on how it works](https://blog.codingconfessions.com/p/compiling-python-to-run-anywhere).

Once compiled, you can then build the included example app and test it from the command line. Here's an example using 
[Kokoro TTS](https://github.com/muna-ai/muna-predictors/blob/main/text-to-speech/kokoro.py) 🔊:

![kokoro example](media/kokoro.mp4)

___

## Useful Links
- [Check out several AI models we've tested](https://github.com/muna-ai/muna-predictors).
- [Join our Slack community](https://muna.ai/slack).
- [Check out our docs](https://docs.muna.ai).
- Learn more about us [on our blog](https://blog.muna.ai).
- Reach out to us at [hi@muna.ai](mailto:hi@muna.ai).

Muna is a product of [NatML Inc](https://github.com/natmlx).
