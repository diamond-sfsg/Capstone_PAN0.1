from __future__ import annotations

import sys
from pathlib import Path

# ===== Add src to PYTHONPATH =====
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]   # PAN_PURPOSE0.1/
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


import pandas as pd

from assign_bin.config import DATA_CLEAN_V2_DIR
from assign_bin.report import print_chunk_report


def main() -> None:
    path = DATA_CLEAN_V2_DIR / "unified_chunks_v2.csv"
    print(f"[INFO] Reading: {path}")

    df = pd.read_csv(path)
    print_chunk_report(df)


if __name__ == "__main__":
    main()