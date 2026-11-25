from pathlib import Path
import hashlib
import subprocess
import sys

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
CSV_FILENAME = "ddr5-and-nvme-prices-2026-07-21-to-2026-08-18.csv"
EXPECTED_SHA256 = "9fb8f539593961e0fcdabcd21a141e3cd53ca3184bf3260da399973fdb594670"

# Primary source: Hugging Face dataset repo (no credentials required)
HF_REPO = "tri-trg/ddr5-nvme-prices-2026"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{CSV_FILENAME}"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file using chunked streaming."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_file(file_path: Path) -> bool:
    """Check if the file exists and matches the expected SHA-256 checksum."""
    if not file_path.exists():
        return False
    actual_hash = compute_sha256(file_path)
    if actual_hash.lower() == EXPECTED_SHA256.lower():
        print(f"[OK] SHA-256 verified: {actual_hash}")
        return True
    print(f"[ERROR] Hash mismatch!\n  Expected: {EXPECTED_SHA256}\n  Actual:   {actual_hash}")
    return False

# ---------------------------------------------------------------------------
# Primary fetcher — Hugging Face (no credentials)
# ---------------------------------------------------------------------------

def _download_hf(dest: Path) -> None:
    """Stream CSV from Hugging Face Datasets to dest."""
    print(f"Downloading from Hugging Face ({HF_REPO})...")
    with requests.get(HF_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
    print()

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_FILENAME

    if verify_file(csv_path):
        print(f"Dataset already present and verified at {csv_path}")
        return

    try:
        _download_hf(csv_path)
    except requests.RequestException as e:
        print(f"[ERROR] Hugging Face download failed: {e}")
        print(
            "\nFallback: place the CSV manually at:\n"
            f"  {csv_path.resolve()}\n"
            "then re-run this script to verify the SHA-256."
        )
        sys.exit(1)

    if not verify_file(csv_path):
        csv_path.unlink(missing_ok=True)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Legacy fetcher — Kaggle (original source, now removed)
# ---------------------------------------------------------------------------

_LEGACY_DATASET_SLUG = "costinflation/ddr5-and-nvme-prices-raw-dataset-2026"

def _legacy_kaggle_main() -> None:
    """
    Original Kaggle downloader kept for reference.
    The dataset was removed from Kaggle; this will fail.
    Use main() instead.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_FILENAME

    if verify_file(csv_path):
        print(f"Dataset already present and verified at {csv_path}")
        return

    print(f"Downloading {_LEGACY_DATASET_SLUG} to {RAW_DIR}...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", _LEGACY_DATASET_SLUG,
        "-p", str(RAW_DIR),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[ERROR] 'kaggle' CLI not found. Ensure the 'ddr5-nvme' conda env is active.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle download failed (exit code {e.returncode}). Check ~/.kaggle/kaggle.json.")
        sys.exit(1)

    if not verify_file(csv_path):
        sys.exit(1)

if __name__ == "__main__":
    main()
