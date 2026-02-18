---
title: Linear Regression
marimo-version: 0.19.11
width: full
sql_output: native
---

```python {.marimo name="setup"}
from dataclasses import dataclass

import torch
import marimo as mo

from torch import nn
```

"From the ground up" is a series of short, technical blog posts about deep learning. This looks like "yet another deep learning from scratch" series, and partly it is - but it has a narrower focus on the abstractions (from `DataLoader`s to `Trainer`s) and performance. The goal of the series is, in other words, to understand the API design and how things are implemented, more so than the maths. This is chapter one, were we train a linear regression using stochastic gradient descent.
<!---->
# Linear Regression
<!---->
## Generate Data
<!---->
We start with a simple function to generate random data, sampling from a multivariate gaussian distribution.

```python {.marimo name="make_regression"}
def make_regression(
    num_samples,
    num_features,
    noise=0.1,
    generator=None,
):
    """Generates a classification problem dataset with gaussian features and a target."""
    X = torch.randn(
        (num_samples, num_features),
        generator=generator,
    )
    y = torch.randn(
        size=(num_samples, 1),
        generator=generator,
    )

    return X * noise, y
```

```python {.marimo}
gen = torch.Generator()
_ = gen.manual_seed(42)

NUM_SAMPLES = 4096
NUM_FEATURES = 6

X, y = make_regression(NUM_SAMPLES, NUM_FEATURES, generator=gen)
```

## The simplest, "imperative" implementation
<!---->
Before we even start subclassing `torch.nn.Module`, let's get down to the basics of linear regression one more time.

We initialise a set of weights and the bias. The shape of the weights is the same as the number of features (or covariates, or exogenous regressors, depending on where you come from). The bias has just one.

```python {.marimo}
w = torch.randn(size=(NUM_FEATURES, 1), requires_grad=True)
b = torch.randn(size=(1,), requires_grad=True)
```

We wrap the parameters in a tuple to operate on them at once, for example to count them:

```python {.marimo}
params = (w, b)
sum(p.numel() for p in params)
```

We then implement the prediction function and the loss:

```python {.marimo}
def predict(X, w, b):
    return X @ w + b


def mean_squared_error(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
```

And finally the training loop. It is comprised of four steps:

1. Set the gradients to zero. In this way, gradients don't accumulate (i.e., add up epoch after epoch).
2. Compute the predictions.
3. Compute the loss.
4. Compute the gradients on the whole model graph.
5. Update the parameters by subtracting the gradient multiplied by a learning rate.

Repeat this step for as many epochs as you like.

```python {.marimo}
EPOCHS = 10
LEARNING_RATE = 0.1

result = {
    "losses": [],
    "params": [],
}

for i in range(EPOCHS):
    for p in params:
        p.grad = None

    y_pred = predict(X, w, b)

    loss = mean_squared_error(y_pred, y)
    result["losses"].append(loss)

    loss.backward()

    for p in params:
        # note that we use `.data`!
        p.data -= LEARNING_RATE * p.grad

    result["params"].append(params)
```

# Let's add some abstractions
<!---->
This code works just fine. But, as soon as you change the model, you need to make changes to other parts of the code. For example, say you want to add a second layer to the linear regression (and thus make it a proper multi-layer perceptron, or MLP): now you have to change the signature of `predict`, to accept more parameters. Are you going to pass more parameters? Make the function accept variadic (`*args`) weights? Or are you going to shove all weights in a list?

This isn't wrong, per se, but you wouldn't want to spend more time thinking of these details. Enter classes, as a way to provide a unified API (i.e., interface) for models. In this way, we can also standardise the training loop in an interface, the `Trainer`.

The `Trainer` interface was first used at scale with PyTorch Lightning, and made popular (or, rather, a standard) with HuggingFace's `transformers`. Fortunately for us, PyTorch already offers a primitive for models: the `nn.Module`. We should only write a minimal abstraction for the `Trainer`.

```python {.marimo name="LinearRegression"}
class LinearRegression(nn.Module):
    """Simple implementation of a linear regression."""

    def __init__(self, num_features, generator=None):
        super().__init__()

        self.weights = nn.Parameter(
            torch.randn(size=(num_features, 1), generator=generator),
            requires_grad=True,
        )
        self.bias = nn.Parameter(
            torch.randn(size=(1,), generator=generator),
            requires_grad=True,
        )

    def forward(self, X):
        return X @ self.weights + self.bias
```

PyTorch `Module`s can magically track the parameters (so you can see them in the `__repr__`, or get them with `model.parameters()`). This is achieved by wrapping the tensor that represents the parameter in a `nn.Parameter` class.
<!---->
Another modern API pattern is using a (data)class to represent a set of arguments. The purpose of this interface is to basically give a namespace for a set of parameters, perhaps to make them easiser to understand, and make function signatures shorter.

```python {.marimo name="NaiveTrainerConfig"}
@dataclass
class NaiveTrainerConfig:
    """Contains the most basic hyperparameters for a training run."""

    epochs: int
    learning_rate: float
```

This is quite shallow, but we are going to add mini-batch size, decay, checkpointing, and more. Another abstraction that HuggingFace's `transformers` provides is the `TrainOutput`, which is simply a `NamedTuple` for the current training step, loss and metrics. We make it a bit different (storing losses and parameters, currently) just to display the results.

```python {.marimo name="NaiveTrainerResult"}
@dataclass
class NaiveTrainerResult:
    """Contains the results of a training run."""

    losses: list[float]
    parameters: list[tuple[nn.Parameter, ...]]
```

```python {.marimo name="NaiveTrainer"}
class NaiveTrainer:
    """The most basic implementation of a training loop.

    No mini batches, no validation set.
    """

    def __init__(
        self,
        model: nn.Module,
        loss,
        X,
        y,
        config: NaiveTrainerConfig,
    ):
        self.model = model
        self.config = config
        self.loss = loss
        self.X = X
        self.y = y

    def train(self) -> NaiveTrainerResult:
        losses = []
        parameters = []

        for i in range(self.config.epochs):
            for param in self.model.parameters():
                param.grad = None

            prediction = self.model(self.X)

            loss = self.loss(prediction, self.y)
            losses.append(loss.item())

            loss.backward()

            for param in self.model.parameters():
                param.data -= self.config.learning_rate * param.grad

            parameters.append(
                tuple(p.data.clone() for p in self.model.parameters())
            )

        return NaiveTrainerResult(losses=losses, parameters=parameters)
```

Having the `__init__` accept an `X` and `y` parameter might feel odd, especially to functional purists. However, we should keep in mind that the `Trainer` is an *orchestrator* of the training itself, that needs to set up some stuff before the training even happens. The `.train()` method should only launch the training, and the preparation steps (we will add progressively more in future notebooks) must be completed before the run starts.

This is especially true in case of distributed training, which implies that data should be sent to a mesh of devices (GPUs). And, moreover, all training frameworks (think of Huggingface `transformers` or PyTorch's `torchtitan`) use this pattern, so we should stick to what people are used to and will encounter.
<!---->
## The training loop, again

```python {.marimo}
model = LinearRegression(NUM_FEATURES)

config = NaiveTrainerConfig(epochs=10, learning_rate=0.1)

trainer = NaiveTrainer(
    model=model,
    loss=mean_squared_error,
    X=X,
    y=y,
    config=config,
)

results = trainer.train()
```

```python {.marimo}
results
```