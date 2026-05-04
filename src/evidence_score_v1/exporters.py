from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_report(df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
