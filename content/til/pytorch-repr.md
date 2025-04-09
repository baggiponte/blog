---
title: "PyTorch Model Visualisation"
date: 2025-04-09
draft: false
description: "a description"
tags: ["example", "tag"]
---

When we load a model from HuggingFace, we might do something like this:

```python
from transformers import AutoModelForCausalLM

model_name = "gpt2"

AutoModelForCausalLM.from_pretrained(model_name)
```

In an interactive environment like ~~Jupyter~~ marimo, we will get a nice representation:

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

This is also known as the (Python object's) `repr`, which is lingo for "string representation". But where does this come from?

Turns out, this is all implemented down at the PyTorch level. If we do this:

```python
import torch.nn as nn

nn.Linear(10, 10)
```

We will get a nice representation:

```
Linear(in_features=10, out_features=10, bias=True)
```

This works for every object in PyTorch that inherits from `nn.Module`. The HuggingFace model is just leveraging this under the hood.
If we want to build a more sophisticated model, we can just do:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


MyModel()
```

And we will get a nice representation:

```
MyModel(
  (linear): Linear(in_features=10, out_features=10, bias=True)
)
```

I haven't looked at the implementation yet, but it looks like that every attribute in the model that is a `nn.Module` will be displayed in the repr.

We can also do this for a `nn.Sequential` model:

```python
import torch.nn as nn

nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
```

```
Sequential(
  (0): Linear(in_features=10, out_features=10, bias=True)
  (1): Linear(in_features=10, out_features=10, bias=True)
)
```

If we want to name the layers, we can do this:

```python
import torch.nn as nn
from collections import OrderedDict

layers = OrderedDict([
    ('linear1', nn.Linear(10, 10)),
    ('linear2', nn.Linear(10, 10)),
])

nn.Sequential(layers)
```

```
Sequential(
  (linear1): Linear(in_features=10, out_features=10, bias=True)
  (linear2): Linear(in_features=10, out_features=10, bias=True)
)
```
