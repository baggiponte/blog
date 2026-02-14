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

# Linear Regression
<!---->
We implement a naive linear regression in PyTorch, introducing a couple of abstractions while making some simplifications.
<!---->
## Generate Data

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

```python {.marimo}
w = torch.randn(size=(NUM_FEATURES, 1), requires_grad=True)
b = torch.randn(size=(1,), requires_grad=True)
```

To operate on the whole set of parameters, we can use the following trick. This allows us to operate on the whole set of parameters at once.

```python {.marimo}
params = (w, b)
sum(p.numel() for p in params)
```

We then implement the prediction function and the loss

```python {.marimo}
def predict(X, w, b):
    return X @ w + b


def mean_squared_error(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
```

And finally the training loop. It is comprised of four steps:

1. Set the gradients to zero. So gradients don't accumulate.
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
This code works, but we would like to make it a bit modular. Classes are of great help: if we can provide a unified API (i.e., interface) for models, we can also try to standardise the training loop.

Fortunately for us, PyTorch already offers a primitive for models: the `nn.Module`. We should only write a minimal abstraction for the `Trainer`.

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

```python {.marimo name="NaiveTrainerConfig"}
@dataclass
class NaiveTrainerConfig:
    """Contains the most basic hyperparameters for a training run."""

    epochs: int
    learning_rate: float
```

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
