from pathlib import Path
import hashlib
import subprocess
import sys

DATASET_SLUG = "costinflation/ddr5-and-nvme-prices-raw-dataset-2026"
RAW_DIR = Path("data/raw")
CSV_FILENAME = "ddr5-and-nvme-prices-2026-07-21-to-2026-08-18.csv"
EXPECTED_SHA256 = "9fb8f539593961e0fcdabcd21a141e3cd53ca3184bf3260da399973fdb594670"


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


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_FILENAME

    if verify_file(csv_path):
        print(f"Dataset already present and verified at {csv_path}")
        return

    print(f"Downloading {DATASET_SLUG} to {RAW_DIR}...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
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