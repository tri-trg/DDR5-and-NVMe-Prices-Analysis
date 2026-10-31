"""Parse structured attributes out of ``product_name``.

INTERFACE (agreed between Q1 / Q5 / Q6 owners)
==============================================
    from src.parse_titles import parse_titles
    parsed = parse_titles(df)          # new columns only, same index as df
    df = df.join(parsed)

Input : ``df`` needs ``series_id`` and ``product_name``.  NOTHING else is read,
        so no price information can leak into the parsed features (important
        for Q6: normalized_price_amount etc. must never be a feature).

Output columns (all derived from the title text only)
-----------------------------------------------------
brand             str   canonical brand ('Corsair', 'Samsung', ...) or 'unknown'
brand_src         str   'name' (known brand), 'model_line' (e.g. Vengeance -> Corsair),
                        'first_word' (unlisted brand = first word of the title), 'none'
capacity_gb_title float total capacity in GB of the whole listing, as written in the title
                        DDR5: whole kit (2x16GB -> 32); NVMe: drive x pack (500GB 4-Pack -> 2000)
kit_size          Int64 DDR5: number of modules in the kit (1, 2, 4...); <NA> for NVMe
module_gb         float DDR5: GB per module (2x16GB -> 16)
kit_src           str   'explicit_nxm' | 'part_number' | 'channel_word' | 'default_single' | 'none'
pack_count        Int64 multi-pack wording ('2-pack', 'pack of 3', '4 pcs'); 1 if none
is_multipack      bool  pack_count > 1
speed_mtps        Int64 DDR5 speed in MT/s (6000MHz -> 6000, PC5-48000 -> 6000)
cas_latency       Int64 DDR5 CL (CL30 -> 30)
form_factor       str   DDR5: 'dimm' | 'so-dimm' | 'rdimm' | 'unknown'
                        NVMe: 'm.2' | 'u.2' | 'e1.s' | 'add-in-card' | 'unknown'
nvme_size         Int64 NVMe M.2 length code (2280, 2230, ...)
pcie_gen          Int64 NVMe PCIe generation (3, 4, 5)
max_mbps          Int64 NVMe: largest 'xxxx MB/s' figure in the title
has_heatsink      bool
has_rgb           bool
kit_total_conflict bool DDR5: the 'NxM GB' total disagrees with another capacity
                        written in the same title

Every field is NA / 'unknown' when the title does not say it. Parsing is done
once per unique (series_id, product_name), so it is fast (~1,200 titles).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Brands
# --------------------------------------------------------------------------- #
# (canonical name, regex). Among the matches found, the one that appears EARLIEST wins.
BRAND_PATTERNS: list[tuple[str, str]] = [
    ("Corsair", r"corsair"),
    ("Kingston", r"kingston"),
    ("Crucial", r"crucial"),
    ("G.Skill", r"g\.?\s?skill"),
    ("TeamGroup", r"team\s?group|t-?force"),
    ("Patriot", r"patriot"),
    ("PNY", r"\bpny\b"),
    ("Samsung", r"samsung"),
    ("SK hynix", r"sk\s?hynix|\bhynix\b"),
    ("Micron", r"micron"),
    ("SanDisk", r"sandisk"),
    ("Western Digital", r"western\s?digital|\bwdc?[\s_-]|\bwdc?\b"),
    ("Seagate", r"seagate|firecuda"),
    ("Lexar", r"lexar"),
    ("Sabrent", r"sabrent"),
    ("Silicon Power", r"silicon\s?power"),
    ("ADATA", r"\badata\b|\bxpg\b"),
    ("Intel", r"^\W*intel\b(?!\s*(?:core|xmp))"),          # only at the start: 'Intel XMP' is not a brand
    ("Solidigm", r"solidigm"),
    ("Kioxia", r"kioxia|toshiba"),
    ("Gigabyte", r"gigabyte|aorus"),
    ("MSI", r"\bmsi\b"),
    ("Acer", r"\bacer\b|predator"),
    ("Netac", r"netac"),
    ("Transcend", r"transcend"),
    ("Mushkin", r"mushkin"),
    ("Biwin", r"biwin"),
    ("Fanxiang", r"fanxiang"),
    ("Orico", r"orico"),
    ("Addlink", r"addlink"),
    ("OLOY", r"\boloy\b"),
    ("Asus", r"\basus\b"),
    ("Zadak", r"zadak"),
    ("Timetec", r"timetec"),
    ("Hikvision", r"hikvision|hiksemi"),
    ("Apacer", r"apacer"),
    ("Acclamator", r"acclamator"),
]
BRAND_PATTERNS = [(b, re.compile(p, re.IGNORECASE)) for b, p in BRAND_PATTERNS]

# Product lines that identify a brand when the brand word itself is missing.
MODEL_LINES: list[tuple[str, str]] = [
    ("Corsair", r"vengeance|dominator|\bmp[56]\d0\b"),
    ("Kingston", r"\bfury\b|renegade|\bnv[23]\b|\bkc3000\b"),
    ("G.Skill", r"trident\s?z|ripjaws|flare\s?x"),
    ("Crucial", r"ballistix"),
    ("Samsung", r"\b9[0-9]0\s?(?:pro|evo)\b"),
    ("Patriot", r"\bviper\b"),
    ("Western Digital", r"wd_?black|\bsn[5-9]\d0x?\b"),
]
MODEL_LINES = [(b, re.compile(p, re.IGNORECASE)) for b, p in MODEL_LINES]

# A brand mentioned only as *compatibility* ("... for Steam Deck, ASUS ROG Ally") is not the seller's brand.
_COMPAT_BEFORE = re.compile(r"(?:\b(?:for|compatible|compatibility|replacement|replace|fits?|works?|with|upgrade|"
                            r"supports?|and|or)\W*|[,/&|+]\s*)$", re.IGNORECASE)

# First-word fallback: words that are NOT brands when they open a title.
_STOP_WORDS = {
    "ssd", "nvme", "m.2", "m2", "pcie", "pci", "internal", "solid", "state", "drive", "drives", "new", "genuine",
    "oem", "oemgenuine", "for", "the", "a", "an", "mac", "macbook", "replacement", "upgrade", "ddr5", "ram",
    "memory", "desktop", "laptop", "computer", "kit", "dual", "mini", "pc", "gaming", "high", "speed", "m", "key",
    "original", "compatible", "sata", "hard", "disk", "hdd", "gen3", "gen4", "gen5", "gen", "up", "to", "with",
    "and", "of", "in", "pack", "set", "3d", "nand", "tlc", "qlc", "sd", "usb", "external", "portable",
    "intel", "amd", "xmp", "expo", "core", "ryzen", "rgb", "black", "white",   # platform names, not memory/SSD brands
}


def _brand_from_first_words(name: str) -> str | None:
    """Unlisted brand: the first 'real' word among the first 3 words, or None."""
    for w in name.split()[:3]:
        clean = re.sub(r"[^\w\-\.&']", "", w).strip(".-")
        low = clean.lower()
        if len(clean) < 2 or low in _STOP_WORDS or re.search(r"\d", clean):
            continue
        return clean.title()
    return None


def _brand(name: str) -> tuple[str, str]:
    best = None
    for brand, rx in BRAND_PATTERNS:
        for m in rx.finditer(name):
            before = name[max(0, m.start() - 35):m.start()]
            if m.start() > 20 and _COMPAT_BEFORE.search(before):
                continue                               # compatibility mention, not the seller
            if best is None or m.start() < best[1]:
                best = (brand, m.start())
            break
    if best:
        return best[0], "name"
    for brand, rx in MODEL_LINES:
        if rx.search(name):
            return brand, "model_line"
    fw = _brand_from_first_words(name)
    if fw:
        return fw, "first_word"
    return "unknown", "none"


# --------------------------------------------------------------------------- #
# Capacity / kit / pack
# --------------------------------------------------------------------------- #
_NUM = r"(\d+(?:\.\d+)?)"
# A capacity number+unit that is not a throughput ("7.4GB/s") and not part of a word.
_CAP = re.compile(_NUM + r"\s*(TB|GB)(?![a-z])(?!\s*/\s*s)", re.IGNORECASE)
# "2x16GB", "2 x 16 GB", "2×16GB", "2*16GB"
_KIT_A = re.compile(r"(?<![\d.])(\d{1,2})\s*[x×*]\s*" + _NUM + r"\s*(TB|GB)(?![a-z])(?!\s*/\s*s)", re.IGNORECASE)
# "16GB x 2", "16GBx2"  (DDR5 only; for NVMe "x4" means PCIe lanes)
_KIT_B = re.compile(_NUM + r"\s*(GB)\s*[x×*]\s*(\d{1,2})(?![\d.])(?!\s*[a-z]*hz)", re.IGNORECASE)
_PART_CT = re.compile(r"\bCT(\d)K(\d{1,3})G", re.IGNORECASE)               # Crucial CT2K32G56C46U5 -> 2 x 32 GB
_PART_CM = re.compile(r"\bCM[A-Z]{1,2}(\d{2,3})GX5M(\d)", re.IGNORECASE)    # Corsair CMK32GX5M2B6000 -> 32 GB, 2 modules
_CH_WORD = re.compile(r"\b(dual|quad|2|4)[\s-]*channel\b|\bkit of (\d)\b|\b(\d)\s*(?:sticks|modules)\b", re.IGNORECASE)
_PACK = [
    # not preceded by a word char: 'x4 PC' (PCIe lanes) and 'Gen4 PC' must not count as 4 packs
    re.compile(r"(?<![\w.])(\d{1,2})\s*[- ]?(?:pack|pk|pcs|ct|count)\b", re.IGNORECASE),
    re.compile(r"\b(?:pack|set|lot|bundle)\s+of\s+(\d{1,2})\b", re.IGNORECASE),
]


def _gb(value: str, unit: str) -> float:
    v = float(value)
    return v * 1000.0 if unit.upper() == "TB" else v


def _pack_count(name: str) -> int:
    for rx in _PACK:
        m = rx.search(name)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 24:
                return n
    return 1


def _plain_caps(text: str) -> list[float]:
    return [_gb(v, u) for v, u in _CAP.findall(text)]


def _parse_ddr5_capacity(name: str) -> dict:
    out = dict(capacity_gb_title=np.nan, kit_size=np.nan, module_gb=np.nan,
               kit_src="none", kit_total_conflict=False)
    m = _KIT_A.search(name)
    span = None
    if m and 1 <= int(m.group(1)) <= 8:
        n, per = int(m.group(1)), _gb(m.group(2), m.group(3))
        out.update(kit_size=n, module_gb=per, capacity_gb_title=n * per, kit_src="explicit_nxm")
        span = m.span()
    else:
        m = _KIT_B.search(name)
        if m and 1 <= int(m.group(3)) <= 8:
            n, per = int(m.group(3)), _gb(m.group(1), m.group(2))
            out.update(kit_size=n, module_gb=per, capacity_gb_title=n * per, kit_src="explicit_nxm")
            span = m.span()

    if span is not None:
        rest = name[:span[0]] + " " + name[span[1]:]
        caps = [c for c in _plain_caps(rest) if 4 <= c <= 1024]
        if caps and out["capacity_gb_title"] not in caps:
            out["kit_total_conflict"] = True
        return out

    caps = [c for c in _plain_caps(name) if 4 <= c <= 1024]
    if not caps:
        return out
    total = caps[0]
    out["capacity_gb_title"] = total
    m = _PART_CT.search(name)
    if m and int(m.group(1)) * int(m.group(2)) == total:
        out.update(kit_size=int(m.group(1)), module_gb=float(m.group(2)), kit_src="part_number")
        return out
    m = _PART_CM.search(name)
    if m and float(m.group(1)) == total and 1 <= int(m.group(2)) <= 8:
        n = int(m.group(2))
        out.update(kit_size=n, module_gb=total / n, kit_src="part_number")
        return out
    m = _CH_WORD.search(name)
    if m:
        word = (m.group(1) or "").lower()
        n = {"dual": 2, "2": 2, "quad": 4, "4": 4}.get(word) or int(m.group(2) or m.group(3))
        if 1 <= n <= 8 and total / n >= 2:
            out.update(kit_size=n, module_gb=total / n, kit_src="channel_word")
            return out
    out.update(kit_size=1, module_gb=total, kit_src="default_single")
    return out


def _parse_nvme_capacity(name: str) -> dict:
    out = dict(capacity_gb_title=np.nan, kit_size=np.nan, module_gb=np.nan,
               kit_src="none", kit_total_conflict=False)
    pack_from_nxm = None
    m = _KIT_A.search(name)
    text = name
    if m and 2 <= int(m.group(1)) <= 8:
        pack_from_nxm = int(m.group(1))
        out["capacity_gb_title"] = _gb(m.group(2), m.group(3))
        return out | {"_pack_nxm": pack_from_nxm}
    caps = [c for c in _plain_caps(text) if 100 <= c <= 16000]
    if caps:
        out["capacity_gb_title"] = caps[0]
    return out


# --------------------------------------------------------------------------- #
# Speed / latency / interface
# --------------------------------------------------------------------------- #
_SPEED = [
    re.compile(r"(?<![\d.])(\d{4,5})\s*(?:mhz|mt/?s|mts)\b", re.IGNORECASE),
    re.compile(r"ddr5[-\s]?(\d{4,5})(?!\d)", re.IGNORECASE),
]
_PC5 = re.compile(r"pc5[-\s]?(\d{5})", re.IGNORECASE)
_CL = [re.compile(r"\bCL\s?-?(\d{2})\b", re.IGNORECASE),
       re.compile(r"\bCAS\s*(?:latency)?\s*(\d{2})\b", re.IGNORECASE),
       re.compile(r"\bC(\d{2})\b")]
_PCIE = [re.compile(r"pc[il][\s-]?(?:e|express)\s*(?:gen\s*)?(\d)(?:\.\d)?", re.IGNORECASE),   # PCIe 4.0, PCI-Express4.0, 'PCle3.0'
         re.compile(r"\bgen\s*(\d)(?:\s*[x×]\s*\d)?", re.IGNORECASE),                       # Gen4x4, Gen 3.0x4
         re.compile(r"pc[il][\s-]?e\s*nvme\s*(\d)\.0", re.IGNORECASE)]                      # 'PCIe NVMe 3.0 x4'
_MBPS = re.compile(r"(\d[\d,]{2,6})\s*MB\s*/\s*s", re.IGNORECASE)
_NVME_SIZE = re.compile(r"\b(2230|2242|2260|2280|22110)\b")


def _speed(name: str):
    for rx in _SPEED:
        for m in rx.finditer(name):
            v = int(m.group(1))
            if 3200 <= v <= 10000:
                return v
    m = _PC5.search(name)
    if m:
        v = int(m.group(1)) // 8
        if 3200 <= v <= 10000:
            return v
    return np.nan


def _cl(name: str):
    for rx in _CL:
        m = rx.search(name)
        if m and 22 <= int(m.group(1)) <= 60:
            return int(m.group(1))
    return np.nan


def _pcie(name: str):
    for rx in _PCIE:
        m = rx.search(name)
        if m and int(m.group(1)) in (3, 4, 5):
            return int(m.group(1))
    return np.nan


def _form_factor(name: str, series_id: str) -> str:
    low = name.lower()
    if series_id == "ddr5_memory":
        if re.search(r"r-?dimm|registered", low):
            return "rdimm"
        if re.search(r"so-?\s?dimm|laptop|notebook", low):
            return "so-dimm"
        if re.search(r"u-?dimm|\bdimm\b|desktop|xmp|expo|heat\s?sink|heat\s?spreader", low):
            return "dimm"            # XMP/EXPO overclocking profiles exist only on desktop modules
        return "unknown"
    if re.search(r"\bu\.?[23]\b", low):
        return "u.2"
    if re.search(r"e1\.s", low):
        return "e1.s"
    if re.search(r"add-?in[\s-]?card|\baic\b", low):
        return "add-in-card"
    if "m.2" in low or "m2" in low or _NVME_SIZE.search(name):
        return "m.2"
    return "unknown"


def _parse_one(name: str, series_id: str) -> dict:
    ddr5 = series_id == "ddr5_memory"
    cap = _parse_ddr5_capacity(name) if ddr5 else _parse_nvme_capacity(name)
    pack_nxm = cap.pop("_pack_nxm", None)
    pack = pack_nxm or _pack_count(name)
    if not ddr5 and pack > 1 and pd.notna(cap["capacity_gb_title"]):
        cap["capacity_gb_title"] = cap["capacity_gb_title"] * pack   # recorded capacity is the pack total
    brand, brand_src = _brand(name)
    m_nv = _NVME_SIZE.search(name)
    mbps = [int(x.replace(",", "")) for x in _MBPS.findall(name)]
    return {
        "brand": brand, "brand_src": brand_src,
        **cap,
        "pack_count": pack, "is_multipack": pack > 1,
        "speed_mtps": _speed(name) if ddr5 else np.nan,
        "cas_latency": _cl(name) if ddr5 else np.nan,
        "form_factor": _form_factor(name, series_id),
        "nvme_size": int(m_nv.group(1)) if (m_nv and not ddr5) else np.nan,
        "pcie_gen": _pcie(name) if not ddr5 else np.nan,
        "max_mbps": max(mbps) if (mbps and not ddr5) else np.nan,
        "has_heatsink": bool(re.search(r"heat\s?sink|heat\s?spreader", name, re.IGNORECASE)),
        "has_rgb": bool(re.search(r"\brgb\b", name, re.IGNORECASE)),
    }


_INT_COLS = ["kit_size", "pack_count", "speed_mtps", "cas_latency", "nvme_size", "pcie_gen", "max_mbps"]
PARSED_COLUMNS = [
    "brand", "brand_src", "capacity_gb_title", "kit_size", "module_gb", "kit_src",
    "pack_count", "is_multipack", "speed_mtps", "cas_latency", "form_factor",
    "nvme_size", "pcie_gen", "max_mbps", "has_heatsink", "has_rgb", "kit_total_conflict",
]


def parse_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Return the parsed columns (see module docstring), index-aligned with ``df``."""
    keys = pd.DataFrame({
        "series_id": df["series_id"].astype(str).to_numpy(),
        "product_name": df["product_name"].astype(str).to_numpy(),
    })
    uniq = keys.drop_duplicates().reset_index(drop=True)
    parsed = pd.DataFrame([_parse_one(n, s) for n, s in zip(uniq["product_name"], uniq["series_id"])])
    uniq = pd.concat([uniq, parsed], axis=1)

    out = keys.merge(uniq, on=["series_id", "product_name"], how="left")
    out = out[PARSED_COLUMNS]
    for c in _INT_COLS:
        out[c] = out[c].astype("Int64")
    for c in ("brand", "brand_src", "kit_src", "form_factor"):
        out[c] = out[c].astype("category")
    out.index = df.index
    return out


# --------------------------------------------------------------------------- #
# Quality checks (plan: "check a sample of 100 titles and report the parse rate")
# --------------------------------------------------------------------------- #
def add_recorded_capacity(df: pd.DataFrame) -> pd.Series:
    """Recorded package capacity in GB (NaN if unresolved). Not a parsed field."""
    q = df["quantity_value"].astype(float)
    u = df["quantity_unit"].astype(str)
    gb = np.where(u == "terabytes", q * 1000.0, np.where(u == "gigabytes", q, np.nan))
    gb = np.where(q == 0, np.nan, gb)
    return pd.Series(gb, index=df.index, name="capacity_gb_recorded")


def _is_parsed(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str) != "unknown")


def parse_report(df: pd.DataFrame) -> pd.DataFrame:
    """Parse rate per field and category. ``df`` must already contain parsed columns."""
    rows = []
    for cat, g in df.groupby("category", observed=True):
        titles = g.drop_duplicates("product_name")
        for col in ["brand", "capacity_gb_title", "kit_size", "speed_mtps", "cas_latency",
                    "pcie_gen", "nvme_size", "max_mbps", "form_factor"]:
            if col in ("kit_size", "speed_mtps", "cas_latency") and cat == "NVMe":
                continue
            if col in ("pcie_gen", "nvme_size", "max_mbps") and cat == "DDR5":
                continue
            ok = _is_parsed(titles[col])
            ok_rows = _is_parsed(g[col])
            rows.append({"category": cat, "field": col,
                         "parsed_titles_%": 100 * ok.mean(), "parsed_rows_%": 100 * ok_rows.mean(),
                         "n_titles": len(titles), "n_rows": len(g)})
    return pd.DataFrame(rows)


def capacity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does the capacity read from the title equal the recorded capacity?"""
    rec = add_recorded_capacity(df)
    ok = rec.notna() & df["capacity_gb_title"].notna()
    match = (rec - df["capacity_gb_title"]).abs() < 0.5
    res = pd.DataFrame({"category": df["category"], "comparable": ok, "match": ok & match})
    t = res.groupby("category", observed=True).agg(rows_comparable=("comparable", "sum"),
                                                   rows_match=("match", "sum"))
    t["match_%"] = 100 * t["rows_match"] / t["rows_comparable"].replace(0, np.nan)
    return t


def unknown_brand_report(df: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """Titles whose brand is unknown, most frequent first words -> extend BRAND_PATTERNS."""
    u = df[df["brand"].astype(str) == "unknown"].drop_duplicates(["category", "product_name"])
    first = u["product_name"].astype(str).str.split().str[0].str.strip(",-()[]").str.title()
    return (pd.DataFrame({"category": u["category"], "first_word": first})
            .value_counts().rename("n_titles").reset_index().head(top))


def sample_for_manual_check(df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Random sample of unique titles with their parsed fields, to inspect by eye."""
    u = df.drop_duplicates(["category", "product_name"])
    parts = []
    for cat in u["category"].dropna().unique():
        g = u[u["category"] == cat]
        parts.append(g.sample(min(len(g), n // 2), random_state=seed))
    s = pd.concat(parts)
    cols = ["category", "product_name"] + [c for c in PARSED_COLUMNS if c not in ("brand_src",)]
    return s[cols].reset_index(drop=True)