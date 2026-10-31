"""Run from the repo root:  python tests/test_cleaning.py   (or: pytest tests/)

Uses real titles copied from the dataset, so no data file is needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cleaning import RowLog, add_flags, analysis_view, check_normalized_price  # noqa: E402

D, N = "ddr5_memory", "nvme_ssd"

# (series, title, capacity resolved?, expected is_system_noise, expected is_nas_noise)
CASES = [
    # --- the 5 complete computers of the data card (unresolved capacity) -> system noise
    (D, "GEEKOM A8 Mini PC, Ryzen 7 8745HS, 16GB DDR5 Upgradeable RAM, 1TB SSD | Creator & Office, USB4, Up to 128GB RAM, 4x 4K Display, 8K, WiFi 6E", False, True, False),
    (D, "MINISFORUM DeskMini UM760 Slim Mini PC AMD Ryzen 5 7640HS Small PC, 16GB DDR5 RAM+512GB PCIe4.0 SSD, HDMI/DP/USB4 Outputs, 2.5G LAN, WiFi 6E", False, True, False),
    (D, 'Dell 15.6" Touchscreen Laptop Computer, 16GB DDR5 RAM, 1TB SSD, Intel Core i5-1334U (10-Core), FHD 1920 x 1080 Display, Windows 11 Pro, Black', False, True, False),
    (D, "Dell Desktop Computers, Intel 20-Core i7-14700, 16GB DDR5 RAM, 1TB PCIe SSD | Dual 4K Support, Remote Work Ready, Windows 11 Pro, 2026 Pro Tower QCT1250", False, True, False),
    (D, "CWWK Firewall Mini PC Intel N Series N100,DDR5 32G RAM 512G NVMe SSD,4 x 2.5GbE i226V LAN,Micro Router Appliance,AES-NI,OPNsense", False, True, False),
    # --- NAS boxes: capacity RESOLVED, so only is_nas_noise catches them
    (D, "Asustor Lockerstor 10 Gen3 AS6810T 10 Bay NAS, AMD Ryzen V3C14 Quad-Core, 16GB ECC DDR5 RAM, Dual 10GbE + Dual 5GbE, 4 M.2 NVMe SSD Slots, USB4 40Gbps, Enterprise Grade for Enthusiasts(Diskless)", True, False, True),
    (D, "TERRAMASTER F8 SSD Plus NAS - 8Bay All SSD NAS Storage Core i3 8-Core 8-Thread CPU, 16GB DDR5 RAM, 10GbE Port, 8 Heat Sinks Included, Palm-Sized Network Attached Storage Peak Performance (Diskless)", True, False, True),
    # --- genuine memory modules must NEVER be flagged, even with tricky wording
    (D, "Crucial 64GB DDR5 RAM, 5600MHz (or 5200MHz or 4800MHz) Laptop Memory Kit, SODIMM 262-Pin, Compatible with 13th Gen Intel Core and AMD Ryzen 6000 - CT2K32G56C46S5", True, False, False),
    (D, "Crucial 64GB DDR5 RAM, 5600MHz Desktop Memory Kit, UDIMM 288-Pin, Compatible with 13th Gen Intel Core and AMD Ryzen 7000 - CT2K32G56C46U5", False, False, False),   # even if unresolved
    (D, "Kingston 32GB DDR5 4800MHz SODIMM Laptop Memory for Synology NAS DS1823xs+", True, False, False),
    (D, "A-Tech 16GB DDR5 5600MHz PC5-44800 CL46 SODIMM 1.1V Non-ECC Unbuffered SO-DIMM 262-Pin Laptop Computer RAM Memory Upgrade Module", False, False, False),
    # --- NVMe: no flags at all
    (N, "Samsung 990 PRO 1TB PCIe 4.0 NVMe M.2 SSD, Compatible with Mini PCs, Windows 11, Intel Core i7", False, False, False),
    (N, "Western Digital 500GB WD Red SN700 NVMe Internal Solid State Drive SSD for NAS Devices - Gen3 PCIe, M.2 2280", True, False, False),
]


def _frame():
    rows = []
    for i, (s, title, resolved, *_rest) in enumerate(CASES):
        rows.append(dict(series_id=s, product_name=title,
                         quantity_value=32.0 if resolved else 0.0,
                         quantity_unit="gigabytes" if resolved else "unknown",
                         price_amount=100.0 + i,
                         normalized_price_amount=(100.0 + i) / 32.0 if resolved else np.nan))
    return pd.DataFrame(rows)


def test_noise_flags():
    out = add_flags(_frame())
    for i, (_, title, _, want_sys, want_nas) in enumerate(CASES):
        assert bool(out["is_system_noise"].iloc[i]) == want_sys, f"is_system_noise wrong for {title[:60]!r}"
        assert bool(out["is_nas_noise"].iloc[i]) == want_nas, f"is_nas_noise wrong for {title[:60]!r}"
        assert bool(out["is_noise"].iloc[i]) == (want_sys or want_nas)


def test_analysis_view_logs_and_removes():
    out = add_flags(_frame())
    log = RowLog()
    v = analysis_view(out, log)
    n_noise = int(out["is_noise"].sum())
    n_unresolved_left = int((out["is_unresolved"] & ~out["is_noise"]).sum())
    assert len(v) == len(out) - n_noise - n_unresolved_left
    assert list(log.table()["rows_removed"]) == [n_noise, n_unresolved_left]
    assert not v["is_noise"].any() and not v["is_unresolved"].any()


def test_recomputed_price_matches_recorded():
    out = add_flags(_frame())
    # DDR5 recorded as price / 32 GB; NVMe 'normalised' per TB is price / 0.032 TB, so only check DDR5 here
    ddr5 = out[out["series_id"].astype(str) == D]
    res = check_normalized_price(ddr5)
    assert res["n_over_tol"] == 0 and res["n_norm_but_unresolved"] == 0 and res["n_blank_norm_but_resolved"] == 0


if __name__ == "__main__":
    test_noise_flags(); test_analysis_view_logs_and_removes(); test_recomputed_price_matches_recorded()
    print(f"OK - {len(CASES)} title cases and 2 view/price checks passed")