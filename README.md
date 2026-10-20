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
* **Methodology & Limitations:** Data collection methods are partially known; limitations include 4,357 unresolved capacity rows and 17 laptop/mini-PC rows mixed into DDR5.

## Research Questions

* **Q1 (Nguyên):** Brand premium/discount percentages per capacity unit (USD/GB for DDR5, USD/TB for NVMe), and rank stability.
* **Q2 (Nguyên):** Median price changes across product × ZIP cohorts between the first 7 days (July 21–27) and last 7 days (Aug 12–18).
* **Q3 (Trí):** Relationship between relative ZIP prices and median household income using joined Census data.
* **Q4 (Trí):** Day-to-day price volatility comparison between DDR5 and NVMe within the same ZIP codes.
* **Q5 (Khải):** Price differences between 2×16 GB kits and 1×32 GB single sticks for 32GB DDR5.
* **Q6 (Khải - ML):** Predicting listing prices of unseen products based on extracted attributes (capacity, brand, speed, form factor) with a target of `log(price_amount)`.

## Key Findings

* to be updated.

## How to Run

1. Recreate the environment:

    ```bash
    conda env create -f environment.yml
    conda activate ddr5-nvme
    ```

2. Download the dataset to `data/raw/` and verify the SHA-256 hash.

3. Execute the notebooks in order: `01_...ipynb`, `02_q1_...`,...
