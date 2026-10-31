# DDR5 and NVMe Prices in 12 U.S. ZIP Markets Analysis (2026)

## Team Members

| Name | Owned Questions | General Tasks |
| :--- | :--- | :--- |
| Chung Nguyên | Q1, Q2 | General EDA, data cleaning (`src/cleaning.py`), slide framework |
| Minh Trí | Q3, Q4 | Repo setup, numerical EDA, Census data join, README |
| Đức Khải | Q5, Q6 | Categorical EDA, title parsing (`parse_titles.py`), ML pipeline |

## Dataset

* **Name:** DDR5 and NVMe Prices Raw Dataset (2026)
* **Source:** CostInflation Team (Kaggle) - [Kaggle link](https://www.kaggle.com/datasets/costinflation/ddr5-and-nvme-prices-raw-dataset-2026)
* **License:** CC0 1.0
* **Timeframe:** July 21 – August 18, 2026 (29 days)
* **Methodology & Limitations:** The team confirms prices are location-specific. Limitations include 4,357 unresolved-capacity rows and 17 laptop/mini-PC rows mixed into DDR5. ZIP-level income is matched to Census ZCTAs, which approximate but are not identical to USPS ZIP Codes; Q3 displays ACS income margins of error but does not propagate them into association intervals.

## Research Questions

* **Q1 (Nguyên):** Brand premium/discount percentages per capacity unit (USD/GB for DDR5, USD/TB for NVMe), and rank stability.
* **Q2 (Nguyên):** Median price changes across product × ZIP cohorts between the first 7 days (July 21–27) and last 7 days (Aug 12–18).
* **Q3 (Trí):** Are higher-income ZIP markets associated with higher DDR5/NVMe prices? Uses 2024 ACS 5-year B19013 median household income and compares same-title/same-date relative prices across ZIP markets; analyzed separately for DDR5 and NVMe.
* **Q4 (Trí):** Day-to-day price volatility comparison between DDR5 and NVMe within the same ZIP codes.
* **Q5 (Khải):** Price differences between 2×16 GB kits and 1×32 GB single sticks for 32GB DDR5.
* **Q6 (Khải - ML):** Predicting listing prices of unseen products based on extracted attributes (capacity, brand, speed, form factor) with a target of `log(price_amount)`.

## Key Findings

* **Q3 (preliminary):** After matching product title and date, ZIP-level adjusted price indices vary by only about −0.046% to +0.036% for DDR5 and −0.018% to +0.041% for NVMe. Income correlations are positive but uncertain across 12 ZIPs (Pearson *r*=0.319 for DDR5 and 0.373 for NVMe; both *p*>0.05). Treat this as a descriptive association, not causal evidence. See [the Q3 notebook](notebooks/02_q3_income_prices.ipynb) for methods, sensitivity checks, and limitations.

## How to Run

1. Recreate the environment:

    ```bash
    conda env create -f environment.yml
    conda activate ddr5-nvme
    ```

2. Download the dataset to `data/raw/` and verify the SHA-256 hash.

    ```bash
    python src/fetch_data.py
    ```

3. Execute the notebooks from the repository root in Jupyter. The initial collection/EDA notebook is `notebooks/01_collection_exploration.ipynb`; Q3 is `notebooks/02_q3_income_prices.ipynb`.

   Q3 retrieves 2024 ACS 5-year income estimates at runtime from the [Census Reporter API](https://api.censusreporter.org/) and requires an internet connection.
