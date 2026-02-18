import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full", sql_output="native")

with app.setup:
    import itertools
    from dataclasses import dataclass

    # type hints
    from collections.abc import Sequence, Iterator, Callable
    from torch.optim import Optimizer
    from torch.nn import Module
    from torch import Generator, Tensor

    import torch
    import torch.nn as nn
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Linear classification
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we move from linear regression to classification. We implement a simple logistic regression using the same PyTorch and our own primitives.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate data
    """)
    return


@app.function
def make_classification(
    num_samples,
    num_features,
    num_classes=2,
    generator: Generator | None = None,
):
    """Generates a classification problem dataset with gaussian features and a target."""
    X = torch.randn(
        (num_samples, num_features),
        generator=generator,
    )
    y = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples, 1),
        generator=generator,
        dtype=torch.float32,
    )

    return X, y


@app.cell
def _():
    gen = torch.Generator()
    _ = gen.manual_seed(42)

    NUM_SAMPLES = 4096 + 1024
    NUM_FEATURES = 6
    NUM_CLASSES = 2

    X, y = make_classification(NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES, gen)
    return NUM_FEATURES, X, gen, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Logistic regression
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since we know how to use `torch.nn.Module`, let's use that already.
    """)
    return


@app.cell
def _(NUM_FEATURES, gen):
    class LogisticRegression(nn.Module):
        def __init__(self, num_features, generator=None):
            super().__init__()
            self.weights = nn.Parameter(
                torch.randn(num_features, 1, generator=generator),
                requires_grad=True,
            )
            self.bias = nn.Parameter(
                torch.randn((1,), generator=generator),
                requires_grad=True,
            )

        def forward(self, X):
            z = X @ self.weights + self.bias
            return torch.sigmoid(z)


    model = LogisticRegression(NUM_FEATURES, generator=gen)
    return (model,)


@app.cell
def _(X, model, trainer, y):
    from linear_regression import NaiveTrainer, NaiveTrainerConfig

    nconfig = NaiveTrainerConfig(epochs=11, learning_rate=0.2)

    ntrainer = NaiveTrainer(
        model=model,
        loss=nn.functional.binary_cross_entropy,
        config=nconfig,
        X=X,
        y=y,
    )

    nresult = trainer.train()
    return (nresult,)


@app.cell
def _(nresult):
    nresult
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # A more accurate Trainer
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The trainer we implemented tracks the loss on the training set. This only measures how the models learns the training data, now how well it "generalises", i.e. how well it could perform on samples that weren't seen during training. A proper `Trainer` should support measuring the loss on a validation set, and also *mini-batches*, i.e. drawing a subset of features to use for the weight update.

    For this, we will introduce a new thin abstraction, the `DataLoader`: a class that accepts some data and is responsible for yielding the batches. We could handle (with some duplication) this logic in the Trainer, but it's good to decouple it from the trainer. Also, PyTorch already offers a `DataLoader` so in this way we can understand how it works.

    Since we follow the semantics of PyTorch, we first need to introduce the `Dataset` class. This is just a thin wrapper that enables an easier manipulation of our features and target. To make this simple, it's basically a wrapper around a set of tensors (in our case) that implements the `__getitem__` method to retrieve samples of `(features, target)` in the dataset. For commodity, we also add the `__len__` method to compute the dataset length, and a quick check to see if all input tensors have the same number of dimensions.
    """)
    return


@app.class_definition
class Dataset:
    def __init__(self, *tensors: Sequence[Tensor]):
        self.tensors = tensors
        lengths = [len(t) for t in tensors]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(
                f"Batch dimension mismatch! Found lengths: {lengths}"
            )

    def __getitem__(self, window: int | slice) -> tuple[torch.Tensor, ...]:
        return tuple(t[window] for t in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


@app.class_definition
@dataclass
class DataLoader:
    dataset: Dataset
    batch_size: int
    shuffle: bool
    generator: Generator | None = None

    def __post_init__(self):
        self.num_samples = len(self.dataset)
        self.num_batches = (
            # to account for incomplete batches
            self.num_samples + self.batch_size - 1
        ) // self.batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self) -> Iterator[tuple[Tensor], ...]:
        if self.shuffle:
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            )
        else:
            indexes = torch.arange(self.num_samples)

        for start_idx in range(0, self.num_samples, self.batch_size):
            # Calculate end_idx, ensuring we don't go out of bounds
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            batch_indexes = indexes[start_idx:end_idx]

            yield self.dataset[batch_indexes]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we can write a new Trainer. We can also make one more improvement: we don't have to write the the optimisation step (the weight update) on our own: we can rely on `torch.optim`, a module that implements abstractions for optimisers. Since optimisers have different parameters, we remove the `learning_rate` parameter from the `TrainerConfig`.
    """)
    return


@app.class_definition
@dataclass
class TrainerConfig:
    num_epochs: int
    shuffle_minibatches: bool
    train_minibatch_size: int
    val_minibatch_size: int | None

    def __post_init__(self):
        if self.val_minibatch_size is None:
            self.val_minibatch_size = self.train_minibatch_size


@app.class_definition
@dataclass
class TrainerRunResult:
    train_losses: list[float]
    val_losses: list[float]


@app.class_definition
@dataclass
class Trainer:
    model: Module
    config: TrainerConfig
    loss: Callable[
        [...], Tensor
    ]  # actually too broad, but writing a Protocol right now feels overkill
    optimizer: Optimizer
    train_dataset: Dataset
    val_dataset: Dataset

    def __post_init__(self):
        """Perform post initialisation options, such as creating the dataloaders."""
        self.train_dataloader = DataLoader(
            self.train_dataset,
            self.config.train_minibatch_size,
            self.config.shuffle_minibatches,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            self.config.val_minibatch_size,
            self.config.shuffle_minibatches,
        )

    def train(self) -> TrainerRunResult:
        train_losses = []
        val_losses = []
        for epoch in range(self.config.num_epochs):
            average_batch_loss = self.train_step()
            train_losses.append(average_batch_loss)

            average_val_loss = self.validation_step()
            val_losses.append(average_val_loss)

        return TrainerRunResult(
            train_losses=train_losses,
            val_losses=val_losses,
        )

    def train_step(self) -> float:
        self.model.train()

        total_train_loss = 0

        for batch in self.train_dataloader:
            # zero the gradients at the beginning of every batch
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)

            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
        return total_train_loss / len(self.train_dataloader)

    def validation_step(self) -> float:
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self._compute_loss(batch)
                total_val_loss += loss.item()

        return total_val_loss / len(self.val_dataloader)

    def _compute_loss(self, batch) -> Tensor:
        features, labels = batch
        predictions = self.model(features)
        return self.loss(predictions, labels)


@app.cell
def _(X, model, y):
    TRAIN_VAL_CUTOFF = 4096
    train_dataset = Dataset(X[:TRAIN_VAL_CUTOFF], y[:TRAIN_VAL_CUTOFF])
    val_dataset = Dataset(X[TRAIN_VAL_CUTOFF:], y[TRAIN_VAL_CUTOFF:])

    config = TrainerConfig(
        num_epochs=10,
        shuffle_minibatches=True,
        train_minibatch_size=32,
        val_minibatch_size=64,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss=nn.functional.binary_cross_entropy,
        optimizer=optimizer,
    )
    return (trainer,)


@app.cell
def _(trainer):
    result = trainer.train()
    return (result,)


@app.cell
def _(result):
    result
    return


if __name__ == "__main__":
    app.run()
