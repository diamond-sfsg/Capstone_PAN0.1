from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_scores_csv(df: pd.DataFrame, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)