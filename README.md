# Muna for Python

![Muna logo](https://raw.githubusercontent.com/muna-ai/.github/main/logo_wide.png)

Compile and run AI models anywhere.

## Installing Muna
Muna is distributed on PyPi. This distribution contains both the Python client and the command line interface (CLI). Run the following command in terminal:
```sh
# Install Muna
$ pip install --upgrade muna
```

> [!NOTE]
> Muna requires Python 3.11+

## Running a Model
First, create a Muna client, specifying your access key ([create one here](https://muna.ai/settings/developer)):

```py
from muna import Muna

# 💥 Create an OpenAI client
openai = Muna("<ACCESS KEY>").beta.openai
```

Next, run a model:
```py
# 🔥 Create a chat completion
completion = openai.chat.completions.create(
  model="@openai/gpt-oss-20b",
  messages=[
    { "role": "user", "content": "What is the capital of France?" }
  ],
)
```

Finally, use the results:
```py
# 🚀 Use the results
print(completion.choices[0].message)
```

___

## Useful Links
- [Check out several AI models we've compiled](https://github.com/muna-ai/muna-predictors).
- [Join our Slack community](https://muna.ai/slack).
- [Check out our docs](https://docs.muna.ai).
- Learn more about us [on our blog](https://blog.muna.ai).
- Reach out to us at [hi@muna.ai](mailto:hi@muna.ai).

Muna is a product of [NatML Inc](https://github.com/natmlx).
