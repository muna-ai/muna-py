# Muna for Python

![Muna logo](https://raw.githubusercontent.com/fxnai/.github/main/logo_wide.png)

Run AI models anywhere.

## Installing Muna
Muna is distributed on PyPi. This distribution contains both the Python client and the command line interface (CLI). To install, open a terminal and run the following command:
```sh
# Install Muna
$ pip install --upgrade muna
```

> [!NOTE]
> Muna requires Python 3.10+

## Retrieving your Access Key
Head over to [muna.ai](https://www.muna.ai/account/developer) to create an account by logging in. Once you do, generate an access key:

![generate access key](https://raw.githubusercontent.com/muna-ai/.github/main/access_key.gif)

## Making a Prediction
First, create a Muna client, specifying your access key:
```py
from muna import Muna

# Create a Muna client
muna = Muna(access_key="<Muna access key>")
```

Then make a prediction:
```py
# Create a prediction
prediction = muna.predictions.create(
    tag="@fxn/greeting",
    inputs={ "name": "Peter" }
)

# Print the returned greeting
print(prediction.results[0])
```

## Using the Muna CLI
Open up a terminal and login to the Muna CLI:
```sh
# Login to Muna
$ muna auth login <ACCESS KEY>
```

Then make a prediction:
```sh
# Make a prediction using the Muna CLI
$ muna predict @fxn/greeting --name Peter
```

___

## Useful Links
- [Discover predictors to use in your apps](https://muna.ai/explore).
- [Join our Slack community](https://muna.ai/slack).
- [Check out our docs](https://docs.muna.ai).
- Learn more about us [on our blog](https://blog.muna.ai).
- Reach out to us at [hi@muna.ai](mailto:hi@muna.ai).

Muna is a product of [NatML Inc](https://github.com/natmlx).
