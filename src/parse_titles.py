"""Title parsing utilities for the DDR5 / NVMe price project (Owner: Đức Khải).

Extracts structured hardware attributes from ``product_name`` without mutating
or dropping any rows, following the design rule in ``src/cleaning.py``:
- Used by Q1 (Chung Nguyên): brand, speed_mhz, form_factor, kit_config, pcie_gen, is_nas_appliance
- Used by Q5 (Đức Khải): kit_config (2x16GB vs 1x32GB), form_factor (UDIMM vs SODIMM), speed_mhz, brand, cas_latency
- Used by Q6 (Đức Khải): feature engineering from product_name with zero price leakage
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Regex patterns & Brand normalisation
# --------------------------------------------------------------------------- #
_SODIMM_PATTERN = re.compile(
    r"(?:SODIMM|SO-DIMM|262-Pin|Laptop|Notebook)",
    flags=re.IGNORECASE,
)

_NAS_PATTERN = re.compile(
    r"(?:\bNAS\b|Network\s+Attached\s+Storage|Diskless)",
    flags=re.IGNORECASE,
)

BRAND_ALIASES: dict[str, str] = {
    "WD_BLACK": "WESTERN DIGITAL",
    "WD": "WESTERN DIGITAL",
    "WESTERN": "WESTERN DIGITAL",
    "SAN": "SANDISK",
    "SANDISK": "SANDISK",
    "G.SKILL": "G.SKILL",
    "GSKILL": "G.SKILL",
    "TEAMGROUP": "TEAMGROUP",
    "TEAM": "TEAMGROUP",
    "KINGSTON": "KINGSTON",
    "SAMSUNG": "SAMSUNG",
    "CRUCIAL": "CRUCIAL",
    "CORSAIR": "CORSAIR",
}


def extract_brand(product_name: str) -> str:
    """Extract and normalise brand name from ``product_name``."""
    clean = re.sub(r"^【.*?】\s*", "", str(product_name)).strip()
    clean = re.sub(r"^\[.*?\]\s*", "", clean).strip()
    if clean.lower().startswith("ddr5") and "acclamator" in clean.lower():
        return "ACCLAMATOR"
    first_token = clean.split()[0].upper().strip(",:;|-") if clean else "UNKNOWN"
    return BRAND_ALIASES.get(first_token, first_token)


def parse_single_title(
    product_name: str,
    series_id: str,
    quantity_value: float = 0.0,
) -> dict:
    """Parse one unique (product_name, series_id, quantity_value) tuple."""
    name = str(product_name).strip()
    brand = extract_brand(name)
    has_rgb = bool(re.search(r"\bRGB\b", name, re.IGNORECASE))
    is_nas = bool(_NAS_PATTERN.search(name))

    if str(series_id) == "ddr5_memory":
        # 1. Kit size: 2x16GB, 16GBx2, 16GB×2, Kit of 2, Single Module, or Part Number
        m_nxm = re.search(r"(\d+)\s*[xX×]\s*(\d+)\s*GB", name, re.IGNORECASE)
        m_mxn = re.search(r"(\d+)\s*GB\s*[xX×]\s*(\d+)", name, re.IGNORECASE)
        m_kit_of = re.search(r"Kit\s+of\s+(\d+)", name, re.IGNORECASE)
        m_single = re.search(r"Single\s+Module", name, re.IGNORECASE)

        if is_nas or quantity_value <= 0:
            kit_modules, module_gb, kit_parse_rule = np.nan, np.nan, "Unresolved_or_System"
        elif m_nxm:
            kit_modules, module_gb = int(m_nxm.group(1)), int(m_nxm.group(2))
            kit_parse_rule = "Explicit_NxM"
        elif m_mxn:
            module_gb, kit_modules = int(m_mxn.group(1)), int(m_mxn.group(2))
            kit_parse_rule = "Explicit_MxN"
        elif m_kit_of:
            kit_modules = int(m_kit_of.group(1))
            module_gb = int(quantity_value // kit_modules)
            kit_parse_rule = "Explicit_KitOfN"
        elif m_single:
            kit_modules, module_gb = 1, int(quantity_value)
            kit_parse_rule = "Explicit_Single"
        elif re.search(r"(?:CT2K|CP2K|GX2|K2-|GK2|DC01)", name):
            kit_modules, module_gb = 2, int(quantity_value // 2)
            kit_parse_rule = "PartNum_Kit2"
        else:
            kit_modules, module_gb = 1, int(quantity_value)
            kit_parse_rule = "Implicit_Single"

        kit_config = (
            f"{int(kit_modules)}x{int(module_gb)}GB"
            if pd.notna(kit_modules)
            else "Unknown"
        )

        # 2. Form factor: check SODIMM before Desktop (handles CMSX32GX5M1A5600C48 title typo)
        if is_nas:
            form_factor = "NAS_Appliance"
        elif _SODIMM_PATTERN.search(name):
            form_factor = "SODIMM"
        else:
            form_factor = "UDIMM"

        # 3. Speed (MHz / MT/s)
        m_spd = re.search(r"(\d{4})\s*(?:MHz|MT/s)", name, re.IGNORECASE)
        speed_mhz = float(m_spd.group(1)) if m_spd else np.nan

        # 4. CAS Latency (CL): explicit CL30/C48 or inside part number (e.g. C46S5, HC38)
        m_cl = re.search(r"\b(?:CL|C)(\d{2})\b", name, re.IGNORECASE)
        if not m_cl:
            m_cl = re.search(r"(?:G\d{2}|H)C(\d{2})(?:[SU]5|[A-Z]{2,3}\d{2})", name)
        cas_latency = float(m_cl.group(1)) if m_cl else np.nan

        pcie_gen = np.nan

    else:  # nvme_ssd
        kit_modules = 1
        module_gb = np.nan
        kit_config = "Single_Drive"
        kit_parse_rule = "NVMe_Drive"
        speed_mhz = np.nan
        cas_latency = np.nan

        # PCIe generation (Gen3, Gen4, Gen5)
        m_gen = re.search(r"(?:PCIe|Gen)\s*([345])(?:\.0|x4|\s*x4)?", name, re.IGNORECASE)
        pcie_gen = float(m_gen.group(1)) if m_gen else np.nan

        # NVMe form factor (M.2 2280, 2230, 2242...)
        m_ff = re.search(r"\b(2280|2230|2242)\b", name)
        if m_ff:
            form_factor = f"M.2_{m_ff.group(1)}"
        elif re.search(r"M\.2", name, re.IGNORECASE):
            form_factor = "M.2_Unspecified"
        else:
            form_factor = "Other_SSD"

    return {
        "brand": brand,
        "kit_modules": kit_modules,
        "module_gb": module_gb,
        "kit_config": kit_config,
        "kit_parse_rule": kit_parse_rule,
        "form_factor": form_factor,
        "speed_mhz": speed_mhz,
        "cas_latency": cas_latency,
        "pcie_gen": pcie_gen,
        "has_rgb": has_rgb,
        "is_nas_appliance": is_nas,
    }


def add_parsed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with parsed title columns merged in.

    Runs regex only on unique ``(product_name, series_id, quantity_value)``
    combinations (~1,180 unique tuples instead of 127,843 rows).
    """
    keys = ["product_name", "series_id", "quantity_value"]
    unique_items = df[keys].drop_duplicates().copy()

    parsed_list = [
        parse_single_title(row.product_name, row.series_id, float(row.quantity_value))
        for row in unique_items.itertuples(index=False)
    ]
    parsed_meta = pd.concat(
        [unique_items.reset_index(drop=True), pd.DataFrame(parsed_list)],
        axis=1,
    )
    return df.merge(parsed_meta, on=keys, how="left")


def audit_ddr5_parse_rate(df_parsed: pd.DataFrame) -> pd.DataFrame:
    """Summary table of parse success rates on DDR5 unique products (Section 9 risk check)."""
    ddr5 = df_parsed[df_parsed["series_id"].astype(str) == "ddr5_memory"].copy()
    if "is_system_noise" in ddr5.columns:
        ddr5 = ddr5[~ddr5["is_system_noise"]]
    if "is_nas_appliance" in ddr5.columns:
        ddr5 = ddr5[~ddr5["is_nas_appliance"]]

    u_all = ddr5.drop_duplicates("product_name")
    u_32 = u_all[u_all["quantity_value"] == 32.0]

    def _rate(sub: pd.DataFrame, cond: pd.Series) -> str:
        n_ok = int(cond.sum())
        total = len(sub)
        return f"{n_ok}/{total} ({n_ok / total * 100:.2f}%)" if total else "0/0"

    rows = [
        ("brand", u_all["brand"].notna() & (u_all["brand"] != "UNKNOWN"),
         u_32["brand"].notna() & (u_32["brand"] != "UNKNOWN")),
        ("kit_config (2x16GB vs 1x32GB)", u_all["kit_config"] != "Unknown",
         u_32["kit_config"].isin(["2x16GB", "1x32GB"])),
        ("form_factor (UDIMM vs SODIMM)", u_all["form_factor"].isin(["UDIMM", "SODIMM"]),
         u_32["form_factor"].isin(["UDIMM", "SODIMM"])),
        ("speed_mhz", u_all["speed_mhz"].notna(), u_32["speed_mhz"].notna()),
        ("cas_latency", u_all["cas_latency"].notna(), u_32["cas_latency"].notna()),
    ]
    return pd.DataFrame([
        {
            "parsed_attribute": attr,
            f"All Pure DDR5 (n={len(u_all)} products)": _rate(u_all, c_all),
            f"DDR5 32GB Q5 (n={len(u_32)} products)": _rate(u_32, c_32),
        }
        for attr, c_all, c_32 in rows
    ])