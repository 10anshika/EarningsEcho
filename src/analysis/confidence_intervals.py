"""
Compute Wilson 95% confidence intervals for directional accuracy metrics.

Grouped by source and sector for all 1d/3d/5d accuracy columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

WINDOWS = ("1d", "3d", "5d")
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Return Wilson score interval bounds for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half_width = z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n) / denom
    return float(center - half_width), float(center + half_width)


def _group_rows(df: pd.DataFrame, by_cols: list[str], group_name: str) -> list[dict]:
    """Build Wilson CI rows for each group/window pair."""
    rows: list[dict] = []
    directional = df[df["signal"] != "NEUTRAL"].copy()
    for group_values, grp in directional.groupby(by_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_payload = dict(zip(by_cols, group_values))
        for w in WINDOWS:
            col = f"correct_{w}"
            valid = grp[col].dropna()
            n = int(len(valid))
            if n == 0:
                continue
            successes = int(valid.sum())
            acc = successes / n
            ci_low, ci_high = wilson_interval(successes, n)
            rows.append(
                {
                    "group_type": group_name,
                    **group_payload,
                    "window": w,
                    "n_directional": n,
                    "n_correct": successes,
                    "accuracy": round(float(acc), 4),
                    "wilson_ci_95_low": round(ci_low, 4),
                    "wilson_ci_95_high": round(ci_high, 4),
                }
            )
    return rows


def run_confidence_intervals(
    csv_path: str | Path = Path("data/backtest_results.csv"),
    out_path: str | Path = Path("data/confidence_intervals_results.csv"),
) -> pd.DataFrame:
    """Compute and save Wilson 95% confidence intervals."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Backtest CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    rows: list[dict] = []
    rows.extend(_group_rows(df, by_cols=["source"], group_name="source"))
    rows.extend(_group_rows(df, by_cols=["sector"], group_name="sector"))
    rows.extend(_group_rows(df, by_cols=["source", "sector"], group_name="source_sector"))

    result_df = pd.DataFrame(rows).sort_values(["group_type", "window", "accuracy"], ascending=[True, True, False])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.success(f"Confidence-interval results saved -> {out_path}")

    print("\n=== Wilson 95% Confidence Intervals (Directional Accuracy) ===")
    print(result_df.head(40).to_string(index=False))
    print(f"\nTotal grouped rows: {len(result_df)}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wilson confidence intervals for EarningsEcho accuracies")
    parser.add_argument("--csv", default="data/backtest_results.csv")
    parser.add_argument("--out", default="data/confidence_intervals_results.csv")
    args = parser.parse_args()
    run_confidence_intervals(csv_path=args.csv, out_path=args.out)
