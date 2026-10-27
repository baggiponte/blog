---
theme: default
layout: cover
highlighter: shiki
colorSchema: light
title: '🐻‍❄️ Polars: Zero to Hero'
---

# 🐻‍❄️ Polars: Zero to Hero
What it is, when to use it, and how to get started

<div class="absolute bottom-10">

    👤 Luca Baggi
    💼 ML Engineer @xtream
    🔮 functime maintainer

</div>

---

# But first, a guessing game

<v-clicks>

> Nowadays, my rule of thumb for pandas is that you should have 5 to 10 times as much RAM as the size of your dataset

[Wes McKinney](https://wesmckinney.com/blog/apache-arrow-pandas-internals/), `pandas`/Apache Arrow creator

> "But first, it’s worth considering *not using pandas* when scaling to large datasets"

`pandas` documentation (until v2.0)

</v-clicks>

---

# 📍 Talk outline

<v-clicks>

### 🐻‍❄️ **What is Polars?**

### 🚀 **What makes Polars so fast?**

### 🧪 **Can Polars be used in production?**

### ❌ **When should I *not* use Polars?**

### ✋ **Question time**

### 📚 **References**

### 📚 **Appendix 1: Small Polars compendium**

### 🔮 **Appendix 2: Intro to functime**

### ✋ **Another round of questions**

</v-clicks>

---

# 🐻‍❄️ What is Polars?
In a nutshell

> Dataframes powered by a multithreaded, vectorized query engine, written in Rust

<v-clicks>

* A `DataFrame` frontend, i.e. work with Python and not a SQL table.
* Utilises all cores on your machine, efficiently (more on this later).
* Has a **query engine** with state-of-the-art algorithms.
* In-process, like `sqlite`.
* Has no other default dependencies (could run in an AWS lambda).

</v-clicks>

---

# 🐻‍❄️ What is Polars?
In plain terms

<v-clicks>

* `pandas` but much faster, no indexes and a more expressive syntax.
  * Not a "drop-in" replacement like `modin` or `cudf` (for the three people out there using it).
* Like `duckdb` but not SQL-first (though you can use SQL too!).

</v-clicks>

---

# 🐻‍❄️ What is Polars?
What it is not

<v-clicks>

* Not a standalone-engine like [`apache/datafusion`](https://datafusion.apache.org/).
* Not a distributed system like [`apache/spark`](https://spark.apache.org/): runs on one node (for now).

</v-clicks>

<v-clicks>

Polars, however, can increase your data processing capabilities so much that you will only need `pyspark` for truly big data, i.e. **complex transformations on more than 1TB**.

> Where the pipeline is simple Polars' streaming mode vastly outperforms Spark and is recommended for all dataset sizes. [Palantir Technologies](https://www.palantir.com/docs/foundry/announcements/#introducing-faster-transforms-for-small-to-medium-sized-datasets)

</v-clicks>

---

# 🚀 What makes Polars so fast?
The key ingredients

<v-clicks>

1. Efficient in-memory representation of the data, following Apache Arrow specification
2. Custom file readers: CSV, parquet, including AWS, HuggingFace...
1. Work stealing, AKA efficient multithreading, thanks to `rayon` and Rust 🦀
1. State-of-the-art algorithms to manipulate data.
2. Extensive optimisations through lazy evaluation.

</v-clicks>

<v-click>

For a thorough introduction by its author, you should check out [this](https://www.youtube.com/watch?v=tqcudsykOGc) and [this](https://www.youtube.com/watch?v=GOOYbl3cqlc) videos.
</v-click>

---

# 🚀 What makes Polars so fast?
Apache Arrow

Arrow is a cross-language specification on how to represent data in memory.

<v-clicks>

1. It's a **columnar memory format** for high-performance analytical queries.
2. Native way to represent missing value (unlike `numpy`).
3. Efficient representation of [strings](https://pola.rs/posts/polars-string-type/) [categorical/enum](https://docs.pola.rs/user-guide/concepts/data-types/categoricals/) data types.
4. Support for **nested datatypes** too: `arrays`, `lists` (arrays of heterogeneous length) and `structs` (dictionaries).

</v-clicks>

---

# 🚀 What makes Polars so fast?
Apache Arrow (cont'd)

<v-clicks>

5. Ensures **zero-copy conversion** between processes that follow the spec
    * For example, you can cast a Polars numeric series to a NumPy array and back without copying the data.
6. Works on CPUs and GPUs.
7. At a lower level, also implements a gRPC-based protocol to transfer data at the wire level (Arrow Flight and Arrow Flight SQL).

</v-clicks>

---

# 🚀 What makes Polars so fast?
Work stealing

<v-clicks>

Polars does not simply implement multithreading. It builds and extends `rayon` to use work-stealing.

The idea is pretty simple: every thread has a queue of tasks to execute. When a thread finishes its task, it steals the next one from the queue.
</v-clicks>

---

# 🚀 What makes Polars so fast?
Extensive optimisations through lazy evaluation

Polars has two modes: lazy and eager (more on this later). In the lazy mode, Polars builds a query plan which can optimise extensively, for example by removing unnecessary expressions as well as branches in the computation and automatically cache bits of data that will be re-used.

---

# 🧪 Is Polars production ready?
Yes.

<v-clicks>

1. Most popular `pandas` alternative (~7M monthly downloads).
    * Unlike other alternatives, is getting traction!
2. Is backed by a **company**, with a clear **product roadmap**: a paid Polars Cloud and, in the future, a distributed engine.
2. Has a broad pool of **maintainers** (>10) beyond the company: QuanSight, edge funds, former JP Morgan employees.
3. NumFOCUS affiliated project.

</v-clicks>

---

# 🧪 Is Polars production ready?
Has a growing ecosystem around it:

<v-clicks>

* Zero-copy conversion to `numpy` and `pytorch` (as well as any other Arrow-compliant library).
* scikit-learn supports Polars DataFrames natively.
* HuggingFace `datasets` can convert data into Polars `DataFrames`.
* Most plotting libraries can work with Polars `DataFrames` (Altair best of all).


</v-clicks>

---

# 🧪 Is Polars production ready?
Has a growing ecosystem around it:

* The `narwhals` project offers a thin compatibility layer to allow developers to write DataFrame-agnostic code, with a subset of the Polars syntax.
* Supported in `pandera` and Dagster.
* Finally shipped v1.0 with a promise of a stable API.

---

# ❌ When not to use Polars
Three cases that come to mind

1. Really big workloads that require Spark.
2. Real time/streaming (e.g. cannot connect to Kafka topics).
3. Excel manipulation and other esoteric file formats.
    * Polars leverages `calamine` to read Excel files, but it might not be as documented as other file formats.

---
layout: intro
---

# ✋Question time

---

# 📚 References

* A small series of [Polars katas](https://github.com/baggiponte/polars-katas), by yours truly.
* [Polars blog](https://pola.rs/posts/) with case studies.

---

# 📚 Appendix 1: Small Polars compendium

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

# 📚 Appendix 1: Small Polars compendium

🪠 I/O but [_blazingly fast_](https://blazinglyfast.party/)

```python{1|3|7}
raw = pl.scan_*("/path/to/source.*") # creates a LazyFrame

raw = pl.scan_parquet("/path/to/*.parquet") # read_parquet works too

processed = raw.pipe(etl, *args, **kwargs)

processed.sink_parquet("path/to/destination.*")
```

---

# 📚 Appendix 1: Small Polars compendium

🪠 What about other formats?

```python
raw = pd.read_*("path/to/source.weird.format")

data = pl.from_pandas(raw)
```

---

# 📚 Appendix 1: Small Polars compendium

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

# 📚 Appendix 1: Small Polars compendium

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

# 📚 Appendix 1: Small Polars compendium

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

# 📚 Appendix 1: Small Polars compendium

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

# 🔮 **Appendix 2: Intro to functime**
Let's jump to another show

<img height="250" width="250"  src="/qr-github.svg">

https://baggiponte.github.io/pydata-global-2023-functime/

---
layout: intro
---

# ✋ Another round of questions

---
layout: intro
---

# 🙏 Thank you!

Please share your feedback!


<div class="absolute right-10 top-10">
<img height="250" width="250"  src="/qr-linkedin.svg">
</div>

<div class="absolute bottom-10">
lucabaggi [at] xtreamers.io | Material will be shared soon
</div>
