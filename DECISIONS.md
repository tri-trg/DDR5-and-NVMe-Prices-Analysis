# Project Decisions & Methodological Log

This document records key technical, data cleaning, and modeling decisions made during the analysis of the DDR5 and NVMe Prices dataset (July 21 – August 18, 2026).

---

## 1. Environment & Data Ingestion

| ID | Date | Owner | Decision | Rationale & Impact |
| :--- | :--- | :--- | :--- | :--- |
| **D-01** | 2026-10-21 | Minh Trí | Standardize environment on Python 3.11 via Conda (`environment.yml`) and verify raw data via `src/fetch_data.py`. | Ensures cross-platform reproducibility across Windows/macOS/Linux and guarantees raw CSV integrity (`SHA-256: 9fb8f539...`) before running pipelines. |
| **D-02** | 2026-10-21 | Team | Retain 10 core columns out of 16 raw columns for downstream analysis. | Drops redundant scraper metadata/URL fields that carry no analytical signal, reducing memory overhead across 127,843 rows. |

---

## 2. Data Cleaning & Filtering Rules (`src/cleaning.py`)

| ID | Date | Owner | Issue / Anomaly | Decision | Affected Rows / Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D-03** | TBD | Chung Nguyên / Minh Trí | Unresolved capacity values (`0.0000`) in raw dataset. | Attempt regex extraction from product title via `src/parse_titles.py`; exclude remaining unresolved rows from unit-price (`USD/GB`, `USD/TB`) calculations. | 4,357 raw rows initially flagged with `0.0000` capacity. |
| **D-04** | TBD | Chung Nguyên / Minh Trí | Non-component hardware (laptops / mini-PCs) misclassified under `ddr5_memory`. | Filter out full-system listings using keyword exclusion rules in `src/cleaning.py` and `src/parse_titles.py`. | 17 identified rows removed to prevent extreme right-skew in DDR5 price distributions. |
| **D-05** | TBD | Đức Khải | Extreme price outliers in 32 GB DDR5 cohorts (Q5). | Evaluate outlier filter sensitivity (e.g., IQR vs. percentile clipping) before comparing `2x16 GB` and `1x32 GB` kits. | Documented in `02_q5_*.ipynb`. |

---

## 3. Feature Engineering, External Data & Modeling

| ID | Date | Owner | Component | Decision | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D-06** | 2026-10-31 | Minh Trí | Q3: ZIP income vs. component prices | Use 2024 ACS 5-Year B19013 median household income and published MOE by five-digit ZCTA; match to price `geography_id` as text. Compare income with the geometric mean of same-title/same-date ZIP price ratios, separately for DDR5 and NVMe. | Prices are location-specific (team confirmed). Treat ZCTA as an approximation of USPS ZIP; only 12 markets support descriptive inference. Income estimates are 2020–2024 in 2024 dollars. ACS values retrieved from the Census Reporter API; MOE is shown but not propagated into association intervals. Analysis and sensitivity checks are in `notebooks/02_q3_income_prices.ipynb`. |
| **D-07** | TBD | Chung Nguyên | Temporal Cohort Windowing (Q2) | Compare first 7 days (July 21–27, 2026) vs. last 7 days (August 12–18, 2026) using median prices. | Mitigates day-of-week scraping noise and resists distortion from single-day promotional spikes. |
| **D-08** | TBD | Đức Khải | ML Target Trans
formation (Q6) | Train regression pipeline on $\log(\text{price\_amount})$ rather than raw USD price; split train/test by product group/title to prevent data leakage across repeated daily snapshots. | Hardware prices exhibit strong positive skewness; log transformation stabilizes variance across capacity tiers. |