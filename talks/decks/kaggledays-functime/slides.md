---
theme: default
layout: cover
highlighter: shiki
colorSchema: light
favicon: >-
  https://raw.githubusercontent.com/neocortexdb/functime/master/docs/img/logo.png
title: 'functime: scalable global forecasting'
---

# 🔮 Forecasting with functime
Scalable forecasting with Polars and global models

<div class="absolute bottom-10">

    👤 Luca Baggi
    💼 ML Engineer @xtream
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

📈 You know what a panel dataset is

⚖️ You work with time series forecasting

</v-clicks>


---

# 📍 Keynote outline

<br>

<v-clicks>

## 🐻‍❄️ Why is Polars so fast?

## 🌏 What is global forecasting?

## 🔮 Dangerous live coding: functime demo!

</v-clicks>


---

# 🐻‍❄️ Why is Polars so fast?
Apache Arrow in-memory representation

<v-clicks>

🪶 A compact representation of in-memory data - i.e. **lower memory footprint**.

📦 Native support for string objects, null values, and **nested datatypes**.

📝 **Zero-copy** between libraries and languages that adopt Arrow.

</v-clicks>


---

# 🐻‍❄️ Why is Polars so fast?
Truly parallel with a powerful query optimiser

<v-clicks>

🎛️ Utilises all **available cores on your machine** thanks to Rust and Rayon (faster than numba!)

🛠️ **Optimises queries** to reduce unneeded memory allocations _and computations_ through Lazy mode.

🌊 Can handle datasets **much larger than RAM** (e.g. streaming execution).

🪺 Great support for **nested datatypes**.

✏️ Has an expressive and elegant **syntax**!

</v-clicks>


---

# 🐻‍❄️ Should I use Polars?
And why should I start tonight?

<v-clicks>

* Compatibile with most of the data and plotting ecosystem.
  * Since Polars adopts the Apache Arrow standard, you can go back and forth with (almost always) zero-copy.
* Polars query engine is [much faster](https://www.pola.rs/benchmarks.html) than anything in-memory.
  * For an overview check out [this keynote](https://www.youtube.com/watch?v=GTVm3QyJ-3I) by its creator, Ritchie Vink.
* Polars ***is* production ready**!
  * It recently crossed 2 million monthly downloads and 20k stars on GitHub.

</v-clicks>


---

# 🌏 What is global forecasting?
And what does it have to do with Polars?

<v-clicks>

* It simply means forecasting multiple time series with a single model.

* In other words, the data looks like a table with (at least) three columns:
  * An identifier for the time series
  * The timestamps
  * The target

* Feature engineering plays a major role
  * And Polars excels at performing these operations quickly.

</v-clicks>


---

# 🌏 What is global forecasting?
Why should I use it?

<v-clicks>

* It proved to be successful to win forecasting competitions (such as the [M5](https://www.sciencedirect.com/science/article/pii/S0169207021001874?via%3Dihub))
* It isn't necessarily better than local forecasting (i.e. fitting one model for each time series), but **definitely is less expensive and time consuming**, especially when it comes to scale.

</v-clicks>


---

# 🌏 What is global forecasting?
A new mindset (?)

Michael Gilliland, formerly at SAS, board member of the International Institute of Forecasting (IIF), [stated](https://blogs.sas.com/content/forecasting/2016/10/25/changing-the-paradigm-for-business-forecasting-part-10/):

> Forecasting is a Huge Waste of Management Time [...]. We need to shift our attention from esoteric model building to the forecasting process itself – its efficiency and its effectiveness.


---
layout: intro
---

# 🔮 functime demo!
## AKA Dangerous live coding

<div class="absolute right-5 top-5">
<img height="150" width="150"  src="/qr-kaggle.svg">
</div>


---
layout: intro
---

# 🙏 Thank you!

Please share your feedback! My address is lucabaggi [at] duck.com

<div class="absolute right-5 top-5">
<img height="150" width="150"  src="/qr-linkedin.svg">
</div>
