---
title: 🎯 Goals
---

# 🎯 Goals
If your expectations are low enough, you won't be disappointed - Mark Twain

<v-click>

🐍 Pythonistas:
</v-click>

<v-clicks>

* Know the most popular libraries used to create Rust + Python libraries.
* Understand a Rust crate directory structure.
* Know how to navigate a Rust + Python project (and, possibly, contribute to it).
* Become acquainted with the Rust syntax.
* Become envious of Rust amazing tooling.
</v-clicks>

<v-click>

🦀 Rustaceans:
</v-click>

<v-clicks>

* Know the most popular libraries used to create Rust + Python libraries.
* Know how to navigate a Rust + Python project.
* Get inspired on how to contribute to the Python ecosystem?
* Don't flex too much.
</v-clicks>

---
layout: two-cols
---

# 😵‍💫 What the 🦀?!
Why don't you [rewrite it in Rust™️](https://unhandledexpression.com/rust/2017/07/10/why-you-should-actually-rewrite-it-in-rust.html)?

<v-clicks>

* A meme since 2017 *at least*.
* Some notable python libraries already have a Rust core:
  * [`orjson`](https://github.com/ijl/orjson) first commit dates to 2018.
  * [`pydantic`](https://github.com/pydantic/pydantic-core) is getting a Rust rewrite.
  * [`ruff`](https://github.com/pola-rs/polars) is a *blazingly fast* linter that just crossed 10.000 ⭐, though its first commit dates to August 2022.
  * [`polars`](https://github.com/pola-rs/polars) a *blazingly fast* dataframe library with more than 15.000 ⭐.

</v-clicks>

::right::

<v-click>

![rust-meme-1](https://i.stack.imgur.com/pQoAq.png)
</v-click>

---

# 🏗️ Building Python + Rust Packages 
Two packaging steps

<v-click>

Not all *oxidised* Python libraries are created equal. For example, `ruff` Python interface is basically a single function that runs the Rust CLI, while `polars` is actually calling Rust code. We have to distinguish two different layers:

</v-click>

<v-clicks>

1. Packaging: making an **distributable artifact** that contains both Rust and Python code.
2. Calling Rust code from Python and viceversa (e.g. [`rust-numpy`](https://github.com/PyO3/rust-numpy) provides a Rust interface for Numpy, and [`tch-rs`](https://github.com/LaurentMazare/tch-rs) offers PyTorch bindings).

</v-clicks>

---

# 🏗️ Building Python + Rust Packages 
The most popular tools

<v-clicks>

* [`maturin`](https://github.com/PyO3/maturin): Python CLI written in Rust for building and publishing Rust-based Python packages. 
  * offers the highest level of abstraction. For a lower level interface, see [`setuptools-rust`](https://github.com/PyO3/setuptools-rust) or [`milksnake`](https://github.com/getsentry/milksnake).
* [`PyO3`](https://github.com/PyO3/pyo3): Rust library to write a native Python module in Rust, or to embed Python in a Rust binary
  * [`rust-cpython`](https://github.com/dgrunwald/rust-cpython) is another option, though less used.

</v-clicks>

---

# 📦 Installation and setup
Minimum supported versions

<v-click>

* Python>=3.6
* Rust>=1.41

</v-click>

<br>

<v-click>

>  💡 *Recommendation: version managers*
>
>  * Rust: [`rustup`](https://rust-lang.github.io/rustup/installation/index.html) is the official version manager.
>  * Python: [`pyenv`](https://github.com/pyenv/pyenv) or [`asdf`](https://github.com/asdf-vm/asdf).
>    * Try out [`rtx`](https://github.com/jdxcode/rtx): it's a - wait for it - Rust rewrite of `asdf`.

</v-click>

--- 

# 📦 Installation and setup
Initialise a project with `maturin`

<v-click>

```bash
pipx install maturin
```

</v-click>

<br>

<v-click>

>  💡 *Recommendation: installing Python CLIs*
>
> Use `pipx` to install Python CLIs. Installation instructions are [here](https://pypa.github.io/pipx/#install-pipx).

</v-click>

<br>

<v-click>

```bash
maturin new pymi-pyo3
```

<br>

* `maturin` will prompt you to choose the bindings you want to use: select `pyo3`.
* This will generate a `pymi-pyo3` folder inside the current one.

</v-click>
