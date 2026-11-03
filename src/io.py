"""Data loading utilities for the DDR5 / NVMe price project.

Import from the repo root, e.g. ``from src.io import load_raw``.
Do NOT add the ``src/`` folder itself to ``sys.path``: this file is called
``io.py`` and would shadow Python's standard-library ``io`` module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CSV_FILENAME = "ddr5-and-nvme-prices-2026-07-21-to-2026-08-18.csv"
RAW_PATH = RAW_DIR / CSV_FILENAME

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
# The 10 columns the team decided to keep (see DECISIONS.md).
KEEP_COLUMNS = [
    "series_id",
    "geography_id",
    "geography_label",
    "observed_date",
    "product_name",
    "quantity_value",
    "quantity_unit",
    "quantity_name",
    "price_amount",
    "normalized_price_amount",
]

# The 6 columns that are constant or redundant (dropped by default).
DROPPED_COLUMNS = [
    "series_title",
    "canonical_url",
    "geography_type",
    "currency_code",
    "normalized_quantity_value",
    "normalized_quantity_unit",
]

ALL_COLUMNS = [
    "series_id", "series_title", "canonical_url", "geography_type",
    "geography_id", "geography_label", "observed_date", "product_name",
    "quantity_value", "quantity_unit", "quantity_name", "price_amount",
    "currency_code", "normalized_price_amount", "normalized_quantity_value",
    "normalized_quantity_unit",
]

# Numbers documented on the Kaggle data card, used to sanity-check the load.
EXPECTED = {
    "n_rows": 127_843,
    "n_columns_raw": 16,
    "rows_by_series": {"ddr5_memory": 18_650, "nvme_ssd": 109_193},
    "n_zip": 12,
    "n_dates": 29,
    "n_product_names": 1_176,
    "n_normalized_nonblank": 123_486,
    "n_unresolved": 4_357,
    "date_min": "2026-07-21",
    "date_max": "2026-08-18",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(
    path: str | Path | None = None,
    columns: str | list[str] = "keep",
    validate: bool = True,
) -> pd.DataFrame:
    """Read the raw CSV with the right dtypes.

    Parameters
    ----------
    path : optional path to the CSV (default: ``data/raw/<CSV_FILENAME>``).
    columns : ``"keep"`` (default, the 10 agreed columns), ``"all"`` (16 columns)
        or an explicit list of column names.
    validate : if True, print a comparison against the Kaggle data card.

    Notes
    -----
    * ``geography_id`` is read as text so ZIP codes keep their leading zeros.
    * Only an empty string is treated as missing. ``"unknown"`` in
      ``quantity_unit`` is a real category and must NOT become NaN.
    * ``observed_date`` is parsed to ``datetime64``.
    """
    path = Path(path) if path else RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/fetch_data.py` from the repo root first."
        )

    if columns == "keep":
        usecols = KEEP_COLUMNS
    elif columns == "all":
        usecols = None
    else:
        usecols = list(columns)

    df = pd.read_csv(
        path,
        usecols=usecols,
        dtype={
            "series_id": "category",
            "geography_id": "string",
            "geography_label": "category",
            "product_name": "string",
            "quantity_unit": "category",
            "quantity_name": "string",
        },
        parse_dates=["observed_date"],
        na_values=[""],
        keep_default_na=False,
    )

    # Keep the column order stable regardless of usecols ordering.
    order = [c for c in ALL_COLUMNS if c in df.columns]
    df = df[order]

    if validate:
        report = validate_raw(df)
        print_validation(report)
    return df


# --------------------------------------------------------------------------- #
# Validation against the data card
# --------------------------------------------------------------------------- #
def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the loaded frame with the numbers on the Kaggle data card.

    Returns a table with columns: check, expected, actual, ok.
    Never raises: a mismatch is reported, not fatal, so that you can investigate.
    """
    rows = []

    def add(name, expected, actual):
        rows.append({"check": name, "expected": expected, "actual": actual,
                     "ok": expected == actual})

    add("n_rows", EXPECTED["n_rows"], len(df))
    for sid, n in EXPECTED["rows_by_series"].items():
        add(f"rows[{sid}]", n, int((df["series_id"] == sid).sum()))
    add("n_zip", EXPECTED["n_zip"], int(df["geography_id"].nunique()))
    add("n_dates", EXPECTED["n_dates"], int(df["observed_date"].nunique()))
    add("n_product_names", EXPECTED["n_product_names"], int(df["product_name"].nunique()))
    add("date_min", EXPECTED["date_min"], str(df["observed_date"].min().date()))
    add("date_max", EXPECTED["date_max"], str(df["observed_date"].max().date()))
    if "normalized_price_amount" in df.columns:
        add("normalized_nonblank", EXPECTED["n_normalized_nonblank"],
            int(df["normalized_price_amount"].notna().sum()))
        add("normalized_blank", EXPECTED["n_unresolved"],
            int(df["normalized_price_amount"].isna().sum()))
    if "quantity_unit" in df.columns:
        add("quantity_unit == 'unknown'", EXPECTED["n_unresolved"],
            int((df["quantity_unit"].astype(str) == "unknown").sum()))
    add("exact duplicate rows", 0, int(df.duplicated().sum()))
    return pd.DataFrame(rows)


def print_validation(report: pd.DataFrame) -> None:
    n_bad = int((~report["ok"]).sum())
    if n_bad == 0:
        print(f"[OK] All {len(report)} checks match the Kaggle data card.")
    else:
        print(f"[WARN] {n_bad}/{len(report)} checks differ from the data card:")
        print(report[~report["ok"]].to_string(index=False))


# --------------------------------------------------------------------------- #
# Saving / loading processed data
# --------------------------------------------------------------------------- #
def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save an intermediate table to ``data/processed/<name>.pkl`` (not committed).

    Pickle keeps dtypes (category, string, datetime) exactly, unlike CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.pkl"
    df.to_pickle(out)
    return out


def load_processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the notebook that creates it first.")
    return pd.read_pickle(path)