# Module 01: Python for Data Science

**Phase 1 — Data Fundamentals** | Est. time: 2–3 months (full-time) · 4–6 months (part-time)

---

## Learning Objectives

- Manipulate large datasets efficiently with NumPy and Pandas
- Create publication-quality visualizations with Matplotlib, Seaborn, and Plotly
- Build interactive dashboards with Streamlit
- Conduct thorough Exploratory Data Analysis (EDA)

---

## Prerequisites

- Module 00: Prerequisites (Python basics + math)

---

## 🐍 Companion Resource

> **[python-for-ai-engineers](https://github.com/MusaIslamFahad/python-for-ai-engineers)** covers NumPy, Pandas, and data science libraries with an AI engineering focus — ideal companion for this module.

---

## Topics Covered

### NumPy
- `ndarray` creation, dtypes, shape, reshape
- Indexing, slicing, fancy indexing, boolean masking, `np.where`
- Broadcasting: vectorized operations without loops
- Linear algebra: `np.dot`, `np.linalg.solve`, `np.linalg.svd`

### Pandas
- Series and DataFrame fundamentals; reading CSV, Excel, JSON, SQL
- Indexing: `loc`, `iloc`, boolean masks
- Data cleaning: `.isnull()`, `.fillna()`, `.dropna()`, `.drop_duplicates()`
- GroupBy, aggregation, `apply`, `transform`, `pivot_table`
- Merge, join, concat; datetime operations and `.dt` accessor

### High-Performance Alternatives
- **Polars**: lazy evaluation, expressions API — 10–100× faster than Pandas for large datasets
- **Dask**: parallel Pandas for data that exceeds RAM

### Visualization
- **Matplotlib**: figure anatomy, subplots, styles, saving
- **Seaborn**: `distplot`, `boxplot`, `heatmap`, `pairplot`
- **Plotly**: interactive charts, scatter, bar, choropleth, 3D

### Applications
- **Streamlit**: turn data scripts into interactive apps in minutes
- **Flask**: lightweight REST APIs for serving data
- Web scraping: `requests`, `BeautifulSoup`, `Selenium` for dynamic content

---

## Quick Reference

```python
import numpy as np, pandas as pd

# NumPy core
a = np.array([1,2,3])
A = np.random.randn(3, 4)
A[A > 0]                        # Boolean mask
np.dot(A, A.T)                  # Matrix multiply

# Pandas core
df = pd.read_csv("data.csv")
df.info(); df.describe()
df.loc[df["age"] > 30, "income"].mean()
df.groupby("city")["spend"].agg(["mean","sum","count"])
merged = pd.merge(df1, df2, on="id", how="left")
```

---

## Project: EDA Dashboard

1. Load a real-world dataset (NYC Taxi, Airbnb, etc.)
2. Clean with Pandas — missing values, dtypes, duplicates
3. Thorough EDA with visualizations
4. Build a Streamlit dashboard with filters and charts
5. Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) (free)

---

## Related Resources

- [Data Science Cheatsheet](../resources/data_science_cheatsheet.md) — NumPy, Pandas quick reference
- 🐍 [python-for-ai-engineers](https://github.com/MusaIslamFahad/python-for-ai-engineers)

**[← Module 00](../00-prerequisites/README.md)** | **[→ Module 02](../02-introduction-to-ml/README.md)**
