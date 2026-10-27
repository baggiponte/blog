---
theme: default
layout: cover
highlighter: shiki
colorSchema: light
favicon: >-
  https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars-logo-dark.svg
title: I know Polars is fast, but my data pipelines are written in pandas!
---

# 🐻‍❄️ I know Polars is fast
... but my data pipelines are written in pandas!

<div class="absolute bottom-10">

    👤 Luca Baggi
    💼 ML Engineer @Futura
    🐍 Organiser @Python Milano

</div>

<div class="absolute right-5 top-5">
<img height="150" width="150"  src="/qr-github.svg">
</div>


---

# 🙋 Raise your hand if...

<br>

<v-clicks>

😎 You know Polars!

🔍 You do exploratory data analysis on a small machine?

🐼 You use pandas in production with {raw,unstructured} data...

🏭 ...or perhaps a distributed system, like PySpark?

🦆 You heard of duckdb?

</v-clicks>


---

# 📍 Keynote outline

<br>

<v-clicks>

## 🏹 pandas is getting faster

## 🐻‍❄️ Why should I use Polars?

## ♻️ Where should I get started?

## ⚡Other features

</v-clicks>


---

# 🏹 pandas is getting faster

<br>

Since v2.0.0, pandas developed a tighter integration with PyArrow, the Python implementation of **Apache Arrow memory format**. What's that?

<v-clicks>

🪶 A compact representation of in-memory data - i.e. **lower memory footprint**.

📦 Native support for string objects, null values, and **nested datatypes**.

📝 **Zero-copy** between libraries and languages that adopt Arrow.

🍁 **Optimised kernels** for data operations thanks to the query engine, Acero.
</v-clicks>


---

# 🏹 pandas is getting faster

<br>

PyArrow will become a required dependency from version 3.0, due next year. This enabled pandas devs to improve a lot on the library:

📽️ [Pandas 2.0 and beyond](https://www.youtube.com/watch?v=NK7RuG4rQpI) (no more "Returning a view versus a copy" 👏)

🔖 [What's new in pandas 2.1?](https://towardsdatascience.com/whats-new-in-pandas-2-1-d26c0b8314a)

<v-click>

To date, only some operations leverage Acero. In other words, **some (most?) operations are still single threaded and unoptimised**.
</v-click>


---

# 🐻‍❄️ Why should I use Polars?

<br>

<v-clicks>

🎛️ Utilises all **available cores on your machine**.

🛠️ **Optimises queries** to reduce unneeded work/memory allocations through Lazy mode.

🌊 Can handle datasets **much larger than RAM** (e.g. streaming execution).

🪺 Great support for **nested datatypes**.

✏️ Has an expressive and elegant **syntax**!

</v-clicks>


---

# 🐻‍❄️ Why should I use Polars?

<br>

* Polars implements Arrow.
  * If you use pandas + Arrow dtypes, you can go back and forth with (almost always) zero-copy.
* Polars query engine is [much faster](https://www.pola.rs/benchmarks.html)
  * For an overview check out [this keynote](https://www.youtube.com/watch?v=GTVm3QyJ-3I) by its creator, Ritchie Vink.
* Polars ***is* production ready**!
  * It recently crossed 2 million monthly downloads.


---

# ♻️ Where should I get started?

<br>

```python
import polars as pd
```

<v-click>

Actually no 🥸
</v-click>


---

# ♻️ Where should I get started?

🪠 I/O

```python{1,6|3,8|4,9}
import pandas as pd

data = pd.read_*("/path/to/source.*")
data.to_*("path/to/destination.*")

import polars as pl # 100% annotated!

data = pl.read_*("/path/to/source.*")
data.write_*("path/to/destination.*")
```


---

# ♻️ Where should I get started?

🪠 I/O but [_blazingly fast_](https://blazinglyfast.party/)

```python{1|3|7}
raw = pl.scan_*("/path/to/source.*") # creates a LazyFrame

raw = pl.scan_parquet("/path/to/*.parquet") # read_parquet works too

processed = raw.pipe(etl, *args, **kwargs)

processed.sink_parquet("path/to/destination.*")
```


---

# ♻️ Where should I get started?

🪠 What about other formats?

```python
raw = pd.read_*("path/to/source.weird.format")

data = pl.from_pandas(raw)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: selection

```python{all|2,3|5|7,8|10,11}
raw.select(
  "col1", "col2"
  pl.col("col1", "col2"),

  pl.col(pl.DataType),      # any valid polars datatype

  pl.col("*"),
  pl.col("$A.*^]"),         # all columns that match a regex pattern

  pl.all(),
  pl.all().exclude(...)     # names, regex, types...
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: selection

```python{all|1|4-6|8,9|11|13}
from polars import selectors as cs

raw.select(
  cs.numeric(),
  ~cs.string(),
  cs.categorical(),

  cs.datetime(time_unit: TimeUnit | Collection[TimeUnit]),
  cs.datetime(time_zone: str | timezone | Collection[str | timezone]),

  cs.contains("substring"),

  cs.all() - cs.numeric(), # use algebra on selectors!
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: manipulate columns

```python{all|5-7|8,9}
(
  questions
  .filter(pl.col("question_times_seen").gt(5)) # also >, >=...
  .with_columns(
    # work with dates
    pl.col("start", "end").dt.day().suffix("_day"),
    pl.col("time_spent").dt.seconds().cast(pl.UInt16).alias("sec"),
    # work with strings
    pl.col("id").str.replace("uuid_", ""),
  )
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: manipulate columns

```python{5-7|8,9}
(
  questions
  .filter(pl.col("question_times_seen").gt(5)) # also >, >=...
  .with_columns(
    # work with arrays!
    pl.col("name").str.split(" ").arr.first().alias("first_name"),
    pl.col("name").str.split(" ").arr.last().alias("last_name"),
    # work with dictionaries
    pl.col("content").struct.field("nested_field")
  )
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: filtering

```python{all|4-7|5|6}
(
  raw
  .sort("simulation_created_at")
  .filter(
    (pl.col("simulation_platform").eq("Medicine"))
    & (pl.count().over("question_uid", "student_uid") == 1)
  )
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: `group_by`

```python{all|3|4-8|5|6|7}
(
  raw
  .group_by("question_uid")
  .agg(
    pl.col("correct", "time_spent").mean().suffix("_mean"),
    pl.col("student_uid").n_unique().shrink_dtype().alias("times_seen"),
    pl.col("question_category_path", "simulation_platform").first(),
  )
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: sliding windows

```python{all|3}
(
  time_series
  .group_by_rolling(index_column="dt", period="2d")
  .agg(
    [
      pl.sum("a").alias("sum_a"),
      pl.min("a").alias("min_a"),
      pl.max("a").alias("max_a"),
    ]
)
```


---

# ♻️ Where should I get started?

🛠️ Data wrangling: window functions

```python{all|4}
(
  time_series
  .with_columns(
    pl.col("c").min().over(["a", "b"]).suffix("_min")
  )
)
```


---

# ⚡Other (cool) features

🌐 Explore the compute graph, or profile the performance

```python{all|1,2|4}
# requires graphviz
raw.filter(...).with_columns(...).show_graph(optimized=True)

result, time = raw.filter(...).with_columns(...).profile()
```


---

# ⚡Other (cool) features

🚀 Even more blazingly-fasterer with `LazyFrame`s

```python{all|1|2|4-9|8}
lazy_frame = pl.scan_*(...)
lazy_frame = pl.from_pandas(...).lazy()

data_frame = (
  lazy_frame
    .filter(...)
    .with_columns(...)
    .collect() # or .sink("path/to/file.parquet")
)
```


---

# ⚡Other (cool) features

🦆 Polars can also quack SQL!

```python{all|5-8|10}
url = "https://gist.githubusercontent.com/ritchie46/cac6b337ea52281aa23c049250a4ff03/raw/89a957ff3919d90e6ef2d34235e6bf22304f3366/pokemon.csv"

pokemon = pl.read_csv(url)

ctx = pl.SQLContext(
  register_globals=True,
  eager_execution=False
)

first_five = ctx.execute("SELECT * from pokemon LIMIT 5")
```


---

# ⚡Other (cool) features

🦆 Polars can also quack SQL _from the command line_!

```bash
$ polars
Polars CLI v0.1.0
Type .help for help.
>> select * FROM read_csv('../../examples/datasets/foods1.csv');
shape: (27, 4)
┌────────────┬──────────┬────────┬──────────┐
│ category   ┆ calories ┆ fats_g ┆ sugars_g │
│ ---        ┆ ---      ┆ ---    ┆ ---      │
│ str        ┆ i64      ┆ f64    ┆ i64      │
╞════════════╪══════════╪════════╪══════════╡
│ vegetables ┆ 45       ┆ 0.5    ┆ 2        │
│ …          ┆ …        ┆ …      ┆ …        │
│ fruit      ┆ 50       ┆ 0.0    ┆ 11       │
└────────────┴──────────┴────────┴──────────┘
```

(but it requires cargo and [building from source](https://github.com/pola-rs/polars-cli) 🥵)!


---

# ⚡Other (cool) features

🦆 Polars can also quack SQL _from the command line_!

```bash
$ echo "SELECT category FROM read_csv('...')" | polars

shape: (27, 1)
┌────────────┐
│ category   │
│ ---        │
│ str        │
╞════════════╡
│ vegetables │
│ …          │
│ fruit      │
└────────────┘
```


---
layout: intro
---

# 🙋 Questions?

<br>

🔗 Check out the [docs](https://pola-rs.github.io/polars/py-polars/html/index.html) or the [user guide](https://pola-rs.github.io/polars-book/user-guide/index.html)!

❓ Ask questions on StackOverflow with the tag [`[python-polars]`](https://stackoverflow.com/questions/tagged/python-polars) or in the (very welcoming) Discord!


---
layout: intro
---

# 🙏 Thank you!

<br>

Please share your feedback! My address is lucabaggi [at] duck.com

<div class="absolute right-5 top-5">
<img height="150" width="150"  src="/qr-linkedin.svg">
</div>
