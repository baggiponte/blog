---
date: '2025-06-15T01:05:26+02:00'
draft: true
title: 'Trackio'
---

Today I found out about a new Gradio (HuggingFace) project: [Trackio](https://github.com/gradio-app/trackio).

It's a ML experiment tracker with a Weights & Biases API. It's really easy to get started:

```bash
uv add -- trackio
```

Then create a Python script:

```python
# say it's named train.py

import random

import trackio as wandb # or just import trackio

def train(
  project: str,
  epochs: int,
  learning_rate: float,
  batch_size: int,
  experiment_name: str | None = None,
):
    wandb.init(
      project=project,
      name=experiment_name,
      config={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
      }
    )

    for epoch in range(epochs):
        train_loss = random.uniform(0.2, 1.0)
        train_acc = random.uniform(0.6, 0.95)

        val_loss = train_loss - random.uniform(0.01, 0.1)
        val_acc = train_acc + random.uniform(0.01, 0.05)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        time.sleep(0.2)

    wandb.finish()

train()
```

This example is self contained and you can run it!

```bash
# start trackio server
uv run trackio show --project={project-name} # use the same name as the run

# launch the run
uv run -- train.py
```

You will see the data being updated in the UI that will pop up in your browser!


