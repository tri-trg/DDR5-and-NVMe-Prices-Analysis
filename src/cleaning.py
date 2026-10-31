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
# Checked against the real data: the 5 titles below (17 rows) are the complete computers in DDR5, e.g.
#   "GEEKOM A8 Mini PC, Ryzen 7 8745HS, 16GB DDR5 ...", "Dell 15.6" ... Laptop Computer, 16GB DDR5 RAM ...",
#   "Dell Desktop Computers, Intel 20-Core i7-14700 ...", "MINISFORUM DeskMini UM760 Slim Mini PC ...",
#   "CWWK Firewall Mini PC Intel N Series N100 ...".
# Plain "laptop" is NOT used on its own because many legitimate SO-DIMM titles say "for Laptop", and
# compatibility mentions such as "AMD Ryzen 7000" or "13th Gen Intel Core" must not match. As a second
# safeguard the flag also requires the row to be unresolved (see add_flags), as in the data card.
_SYSTEM_PATTERN = re.compile(
    r"(?:mini[\s-]?pc|desktop (?:computer|pc)|gaming (?:pc|desktop|laptop)|laptop computer|"
    r"all[\s-]in[\s-]one|notebook computer|\bintel core i[3579]\b|\bryzen\s+[3579]\b|"
    r"windows 1[01]|\bwi-?fi\b|\bnuc\b)",
    flags=re.IGNORECASE,
)

# Wording that only a real memory module has. A title with any of these is never "a complete computer",
# even if it says "Laptop Computer RAM" (e.g. "A-Tech ... SO-DIMM 262-Pin Laptop Computer RAM Memory Upgrade Module").
_MODULE_GUARD = re.compile(
    r"(?:so-?dimm|u-?dimm|\bdimm\b|\bpc5-\d{4,5}|\b\d{3}-pin|ram memory|memory (?:kit|module|upgrade))",
    flags=re.IGNORECASE,
)

# NAS boxes sit in the DDR5 category WITH a resolved capacity (they quote the RAM inside), so the data
# card's 17-row check (which needs an unresolved capacity) does not see them. Real examples:
#   "Asustor Lockerstor 10 Gen3 AS6810T 10 Bay NAS ... 16GB ECC DDR5 RAM ... (Diskless)"  (3 rows)
#   "TERRAMASTER F8 SSD Plus NAS - 8Bay All SSD NAS Storage ... 16GB DDR5 RAM ... (Diskless)"  (8 rows)
# The bare word "NAS" is deliberately NOT enough: a genuine RAM module can say "for Synology NAS".
_NAS_PATTERN = re.compile(r"(?:\b\d+[\s-]?bay\b|\bdiskless\b|network attached storage)", flags=re.IGNORECASE)

# What the full dataset contains (verified on the real CSV); used by check_noise_flags().
NOISE_EXPECTED = {"system_rows": 17, "system_titles": 5, "nas_rows": 11, "nas_titles": 2}

SERIES_SHORT = {"ddr5_memory": "DDR5", "nvme_ssd": "NVMe"}


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with descriptive flag columns. No rows change.

    Added columns
    -------------
    category          'DDR5' or 'NVMe' (short label of series_id)
    is_unresolved     quantity unknown / 0 -> normalised price is blank
    is_system_noise   DDR5 row that looks like a complete laptop/desktop/mini-PC (data card: 17 rows)
    is_nas_noise      DDR5 row that is a NAS box (found by us; resolved capacity, so not in the 17)
    is_noise          is_system_noise or is_nas_noise  (rows that are not memory modules)
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
    names = out["product_name"].astype(str)
    looks_like_system = names.str.contains(_SYSTEM_PATTERN) & ~names.str.contains(_MODULE_GUARD)
    out["is_system_noise"] = is_ddr5 & out["is_unresolved"] & looks_like_system
    out["is_nas_noise"] = is_ddr5 & names.str.contains(_NAS_PATTERN)
    out["is_noise"] = out["is_system_noise"] | out["is_nas_noise"]

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
    drop_noise: bool = True,
) -> pd.DataFrame:
    """Return the rows suitable for price-per-capacity analysis.

    Nothing is modified; rows are only excluded, and each exclusion is logged
    with its reason. Questions that need ALL rows (e.g. Q2 uses price_amount
    only) can pass drop_unresolved=False.
    """
    log = log or RowLog()
    out = df
    if drop_noise:
        out = log.step(
            out, "drop complete-system / NAS rows (DDR5 category noise)",
            lambda d: d[~d["is_noise"]],
            reason="laptop/desktop/mini-PC and NAS listings are not memory modules",
        )
    if drop_unresolved:
        out = log.step(
            out, "drop unresolved capacity",
            lambda d: d[~d["is_unresolved"]],
            reason="quantity unknown -> normalised price blank, cannot compare per GB/TB",
        )
    return out


def check_noise_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the noise flags with what the FULL real dataset contains.

    Only meaningful on the complete data (a ZIP sample has fewer rows). Never raises.
    """
    sysn, nas = df[df["is_system_noise"]], df[df["is_nas_noise"]]
    rows = [
        ("system-noise rows (data card: 17)", NOISE_EXPECTED["system_rows"], len(sysn)),
        ("system-noise titles (data card: 5)", NOISE_EXPECTED["system_titles"], sysn["product_name"].nunique()),
        ("NAS rows", NOISE_EXPECTED["nas_rows"], len(nas)),
        ("NAS titles", NOISE_EXPECTED["nas_titles"], nas["product_name"].nunique()),
        ("DDR5 rows unresolved but NOT flagged", 0,
         int(((df["series_id"].astype(str) == "ddr5_memory") & df["is_unresolved"] & ~df["is_system_noise"]).sum())),
    ]
    t = pd.DataFrame(rows, columns=["check", "expected (full data)", "actual"])
    t["ok"] = t["expected (full data)"] == t["actual"]
    return t


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
        "NAS-noise rows": int(df["is_nas_noise"].sum()) if "is_nas_noise" in df else np.nan,
    }
    return pd.DataFrame({"check": list(checks), "count": list(checks.values())})
