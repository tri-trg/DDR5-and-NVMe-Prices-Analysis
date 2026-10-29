"""Cleaning helpers for the DDR5 / NVMe price project.

Design rule (from the data card): the raw data is NOT corrected or deleted
silently. We (1) add boolean FLAG columns, (2) apply filters explicitly in
each analysis, and (3) log the row count before/after every filter.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Row-count log (the course requires a row count after every data-changing step)
# --------------------------------------------------------------------------- #
class RowLog:
    """Record the row count before/after each step.

    >>> log = RowLog()
    >>> df = log.step(df, "drop unresolved", lambda d: d[~d["is_unresolved"]],
    ...               reason="capacity unknown, cannot normalise price")
    >>> log.table()
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def step(self, df: pd.DataFrame, name: str, func, reason: str = "") -> pd.DataFrame:
        before = len(df)
        out = func(df)
        after = len(out)
        self.records.append(
            {"step": name, "rows_before": before, "rows_after": after,
             "rows_removed": before - after, "reason": reason}
        )
        print(f"[{name}] {before:,} -> {after:,} rows (removed {before - after:,})")
        return out

    def note(self, df: pd.DataFrame, name: str, reason: str = "") -> pd.DataFrame:
        """Log a step that does not change the row count (e.g. adding columns)."""
        return self.step(df, name, lambda d: d, reason)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
# Wording typical of complete computers rather than loose RAM / SSD parts.
# NOTE: plain "laptop" is NOT used on its own because many legitimate DDR5
# SO-DIMM titles say "for Laptop". The flag therefore also requires the row to
# be unresolved (see add_flags), matching the data card's finding that all 17
# noise observations have unresolved capacity.
_SYSTEM_PATTERN = re.compile(
    r"(?:mini[\s-]?pc|desktop (?:computer|pc)|gaming (?:pc|desktop|laptop)|laptop computer|"
    r"all[\s-]in[\s-]one|notebook computer|intel core i[3579]|ryzen [3579]|"
    r"windows 1[01]|\bwi-?fi\b|\bnuc\b)",
    flags=re.IGNORECASE,
)

SERIES_SHORT = {"ddr5_memory": "DDR5", "nvme_ssd": "NVMe"}


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with descriptive flag columns. No rows change.

    Added columns
    -------------
    category          'DDR5' or 'NVMe' (short label of series_id)
    is_unresolved     quantity unknown / 0 -> normalised price is blank
    is_system_noise   DDR5 row that looks like a complete laptop/desktop/mini-PC
    price_per_unit_recomputed
                      price_amount / quantity (USD per GB for DDR5, per TB for NVMe);
                      NaN when capacity unresolved. Used to cross-check the
                      source-recorded normalized_price_amount.
    """
    out = df.copy()
    out["category"] = out["series_id"].astype(str).map(SERIES_SHORT).astype("category")

    qty = out["quantity_value"].astype(float)
    unit = out["quantity_unit"].astype(str)
    out["is_unresolved"] = (unit == "unknown") | (qty == 0)

    is_ddr5 = out["series_id"].astype(str) == "ddr5_memory"
    looks_like_system = out["product_name"].astype(str).str.contains(_SYSTEM_PATTERN)
    out["is_system_noise"] = is_ddr5 & out["is_unresolved"] & looks_like_system

    # Recompute price per target unit (GB for DDR5, TB for NVMe; 1,000 GB = 1 TB).
    qty_gb = np.where(unit == "gigabytes", qty,
                      np.where(unit == "terabytes", qty * 1000.0, np.nan))
    qty_target = np.where(is_ddr5, qty_gb, qty_gb / 1000.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppu = out["price_amount"].astype(float) / qty_target
    ppu = np.where(np.isfinite(ppu) & ~out["is_unresolved"], ppu, np.nan)
    out["price_per_unit_recomputed"] = ppu
    return out


def check_normalized_price(df: pd.DataFrame, tol: float = 0.011) -> dict:
    """Compare recorded normalized_price_amount with our recomputation.

    The data card says they agree within USD 0.011. Returns summary numbers.
    """
    both = df["normalized_price_amount"].notna() & df["price_per_unit_recomputed"].notna()
    diff = (df.loc[both, "normalized_price_amount"] - df.loc[both, "price_per_unit_recomputed"]).abs()
    return {
        "n_compared": int(both.sum()),
        "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
        "n_over_tol": int((diff > tol).sum()),
        "n_blank_norm_but_resolved": int(
            (df["normalized_price_amount"].isna() & ~df["is_unresolved"]).sum()
        ),
        "n_norm_but_unresolved": int(
            (df["normalized_price_amount"].notna() & df["is_unresolved"]).sum()
        ),
    }


def flag_outliers_iqr(
    df: pd.DataFrame,
    col: str,
    by: list[str] | str | None = None,
    k: float = 1.5,
    use_log: bool = True,
) -> pd.Series:
    """Boolean Series: True where ``col`` is outside [Q1 - k*IQR, Q3 + k*IQR].

    Computed within groups ``by`` (e.g. ['series_id']) so that DDR5 and NVMe
    are never judged against each other. ``k`` = 1.5 (usual) or 3 (extreme);
    changing k is a ready-made robustness check (criterion D6).
    """
    x = df[col].astype(float)
    if use_log:
        x = np.log(x.where(x > 0))

    def _bounds(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - k * iqr) | (s > q3 + k * iqr)

    if by is None:
        return _bounds(x).fillna(False)
    return x.groupby([df[b] for b in ([by] if isinstance(by, str) else by)],
                     observed=True).transform(_bounds).fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Analysis views (explicit, logged filters)
# --------------------------------------------------------------------------- #
def analysis_view(
    df: pd.DataFrame,
    log: RowLog | None = None,
    drop_unresolved: bool = True,
    drop_system_noise: bool = True,
) -> pd.DataFrame:
    """Return the rows suitable for price-per-capacity analysis.

    Nothing is modified; rows are only excluded, and each exclusion is logged
    with its reason. Questions that need ALL rows (e.g. Q2 uses price_amount
    only) can pass drop_unresolved=False.
    """
    log = log or RowLog()
    out = df
    if drop_system_noise:
        out = log.step(
            out, "drop complete-system rows (DDR5 category noise)",
            lambda d: d[~d["is_system_noise"]],
            reason="laptop/desktop/mini-PC listings are not memory modules",
        )
    if drop_unresolved:
        out = log.step(
            out, "drop unresolved capacity",
            lambda d: d[~d["is_unresolved"]],
            reason="quantity unknown -> normalised price blank, cannot compare per GB/TB",
        )
    return out


def basic_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-check summary of structural quality (for section 3.2)."""
    checks = {
        "rows": len(df),
        "exact duplicate rows": int(df.duplicated().sum()),
        "fully empty rows": int(df.isna().all(axis=1).sum()),
        "price_amount <= 0": int((df["price_amount"] <= 0).sum()),
        "price_amount missing": int(df["price_amount"].isna().sum()),
        "product_name missing/blank": int(
            (df["product_name"].isna() | (df["product_name"].astype(str).str.strip() == "")).sum()
        ),
        "product_name with leading/trailing spaces": int(
            (df["product_name"].astype(str) != df["product_name"].astype(str).str.strip()).sum()
        ),
        "unresolved capacity": int(df["is_unresolved"].sum()) if "is_unresolved" in df else np.nan,
        "system-noise rows": int(df["is_system_noise"].sum()) if "is_system_noise" in df else np.nan,
    }
    return pd.DataFrame({"check": list(checks), "count": list(checks.values())})
