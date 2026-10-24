---
theme: default
layout: cover
highlighter: prisma
colorSchema: light
favicon: >-
  https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars-logo-dark.svg
title: 🐻‍❄️ Polars
---

# 🐻‍❄️ Polars
Is the great dataframe showdown finally over?

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

🔍 You had to do exploratory work...

🪨 on a lot of {raw,unstructured} data...

💻 on a small machine...

🏭 without {time,budget,resources,expertise} to use a distributed system?
</v-clicks>


---

# 📍 Keynote outline

<br>

<v-clicks>

## 🏎️ pandas 2.0 supports Arrow: why should I use polars?

## ♻️ I get the point: polars is faster, but rewriting code takes time

## ⚡Other features

</v-clicks>


---

# 🏎️ pandas 2.0 and Arrow

<br>

With pandas>=2.0, you can use `dtype_backend=“pyarrow”` to use the Arrow in-memory format. This means:

<v-clicks>

✅ More efficient data types and **lower memory usage**.

❌ _Only some_ operations leverage Acero, i.e. pyarrow query/compute engine (notably, not `groupby`s!). In other words, **some (most?) operations are still single threaded and unoptimised**.
</v-clicks>


---

# 🏎️ pandas 2.0 and Arrow

<br>

On the other hand, Polars:

<v-clicks>

🎛️ Utilises all **available cores on your machine**.

🛠️ **Optimises queries** to reduce unneeded work/memory allocations through Lazy mode.

🌊 Can handle datasets **much larger than RAM** (e.g. streaming execution).

🪺 Great support for **nested datatypes**.
</v-clicks>


---

# 🏎️ pandas 2.0 and Arrow

📹 Must watch

🏹 Alessandro Molina covered you all about Arrow in [Apache Arrow as a full stack data engineering solution](https://pycon.it/en/event/apache-arrow-as-a-full-stack-data-engineering-solution).
<br>

🔥 Alberto Danese made a performance comparison in [Beyond Pandas: lightning fast in-memory dataframes with Polars](https://pycon.it/en/event/beyond-pandas-lightning-fast-in-memory-dataframes-with-polars).

🐼 Fabio Lipreri will show you how to speed up your pandas pipelines [pandas on steroids](https://pycon.it/en/event/pandas-on-steroids) in Room Pizza (this room), 16:30.


---

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

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

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

🪠 I/O but [_blazingly fast_](https://blazinglyfast.party/)

```python{1|3|7}
raw = pl.scan_*("/path/to/source.*") # creates a LazyFrame

raw = pl.scan_parquet("/path/to/*.parquet") # read_parquet works too

processed = raw.pipe(etl, *args, **kwargs)

processed.sink_parquet("path/to/destination.*")
```


---

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

🪠 What about other formats?

```python
raw = pd.read_*("path/to/source.weird.format")

data = pl.from_pandas(raw)
```


---

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

🛠️ Data wrangling: selection

```python{1,9|2|3|4|5|6|7|8}
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

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

🛠️ Data wrangling: manipulate columns

```python{all|4,15|5-7|8-9|10-12|13-14}
(
  questions
  .filter(pl.col("question_times_seen").gt(5)) # also >, >=...
  .with_columns(
    # work with dates
    pl.col("start", "end").dt.day().suffix("_day"),
    pl.col("time_spent").dt.seconds().cast(pl.UInt16).alias("sec"),
    # work with strings
    pl.col("id").str.replace("uuid_", ""),
    # work with arrays!
    pl.col("name").str.split(" ").arr.first().alias("first_name"),
    pl.col("name").str.split(" ").arr.last().alias("last_name"),
    # work with dictionaries
    pl.col("content").struct.field("nested_field")
  )
)
```


---

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

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

# ♻️ Rewriting pandas code ~~takes time~~ is easy!

🛠️ Data wrangling: `groupby`

```python{all|3|4-8|5|6|7}
(
  raw
  .groupby("question_uid")
  .agg(
    pl.col("correct", "time_spent").mean().suffix("_mean"),
    pl.col("student_uid").n_unique().shrink_dtype().alias("times_seen"),
    pl.col("question_category_path", "simulation_platform").first(),
  )
)
```

And it works for up- and down-sampling date types too (temporal aggregation)!


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
lazy_frame: pl.LazyFrame = pl.scan_*(...)
lazy_frame: pl.DataFrame = pl.from_pandas(...).lazy()

data_frame: pl.DataFrame = (
  lazy_frame
    .filter(...)
    .with_columns(...)
    .collect() # or .sink("path/to/file.parquet")
)
```


---

# ⚡Other (cool) features

🦆 Polars can also quack SQL!

```python{1|3,4|6|8}
ctx = pl.SQLContext()

df = pl.DataFrame({"a": [1, 2, 3]})
lf = pl.LazyFrame({"b": [4, 5, 6]})

ctx = pl.SQLContext(register_globals=True)

ctx = pl.SQLContext(df=df, lf=lf)
```


---

# ⚡Other (cool) features

🦆 Polars can also quack SQL!

```python{3-8|5,6|8}
url = "https://gist.githubusercontent.com/ritchie46/cac6b337ea52281aa23c049250a4ff03/raw/89a957ff3919d90e6ef2d34235e6bf22304f3366/pokemon.csv"

pokemon = pl.read_csv(url)

# set to False to optimise your queries
ctx = pl.SQLContext(register_globals=True, eager_execution=True)

first_five = ctx.execute("SELECT * from pokemon LIMIT 5")
```


---

# ⚡Other (cool) features

🦆 Polars can also quack SQL _from the command line_

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

(but it requires cargo and [building from source](https://github.com/pola-rs/polars/tree/main/polars-cli) 🥵)!


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

🔗 Check out the [docs](https://pola-rs.github.io/polars/py-polars/html/index.html) or the [user guide](https://pola-rs.github.io/polars-book/user-guide/index.html)!


---
layout: intro
---

# 🙏 Thank you!

Please share your feedback! My address is lucabaggi [at] duck.com

<div class="absolute right-5 top-5">
<img height="150" width="150"  src="/qr-linkedin.svg">
</div>
