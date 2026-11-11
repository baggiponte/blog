---
theme: ..
layout: cover
title: the slides theme
event: pycon italia
date: may 2027
---

# the slides theme

the blog's paper, type and code blocks, on a slide

::aside::

<div class="qr">qr code here</div>

<style>
.qr {
  width: 140px;
  height: 140px;
  display: grid;
  place-items: center;
  border: 1px dashed var(--rule);
  border-radius: 12px;
  font: 12px var(--mono);
  color: var(--muted);
}
</style>

---

# what you get

a subtitle in the muted ink

- the blog's ivory paper and warm accent, in light and dark
- Bricolage Grotesque headings, Instrument Sans text
- <mark>highlighted words</mark> and `inline code`, as in posts
- [links](https://baggiponte.com) underlined in the accent

> a quote sits on an accent rule, in the muted ink.

---
layout: section
kicker: part 1
---

# a section divider

---

# code, as in posts

```python [lazy.py]
import polars as pl

df = (
    pl.scan_parquet("trips.parquet")
    .filter(pl.col("distance") > 0)
    .group_by("vendor")
    .agg(pl.col("fare").mean())
)
```

the filename in square brackets becomes the caption.

---
layout: code
---

# the code layout

```rust [src/lib.rs]{2-4}
#[pyfunction]
fn fib(n: u64) -> u64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

#[pymodule]
fn fastfib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fib, m)?)
}
```

---
layout: two-cols
---

# two columns

the built-in layouts pick up the theme

- left column
- with a list

::right::

| library | lazy |
| ------- | ---- |
| pandas  | no   |
| polars  | yes  |
| duckdb  | yes  |

---
layout: section
footer: false
---

# thank you
