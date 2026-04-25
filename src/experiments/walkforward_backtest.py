"""
Walk-forward percentile backtest to reduce look-ahead bias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))
from config.settings import PERCENTILE_HIGH, PERCENTILE_LOW


def _assign_signal(score: float, high: float, low: float) -> str:
    """Assign tri-state signal for one score."""
    if score >= high:
        return "NEGATIVE"
    if score <= low:
        return "POSITIVE"
    return "NEUTRAL"


def run_walkforward_backtest(
    out_path: str | Path = _ROOT / "data" / "walkforward_results.csv",
) -> tuple[pd.DataFrame, float]:
    """Run expanding-window P80/P20 validation on 2025+ rows only."""
    df = pd.read_csv(_ROOT / "data" / "backtest_results.csv")
    df["call_date"] = pd.to_datetime(df["call_date"])
    df = df.sort_values("call_date").reset_index(drop=True)

    eval_df = df[df["call_date"] >= pd.Timestamp("2025-01-01")].copy()
    rows: list[dict] = []

    for idx, row in eval_df.iterrows():
        history = df[df["call_date"] < row["call_date"]]
        if len(history) < 20:
            continue

        high = float(np.percentile(history["ew_risk_score"], PERCENTILE_HIGH))
        low = float(np.percentile(history["ew_risk_score"], PERCENTILE_LOW))
        signal = _assign_signal(float(row["ew_risk_score"]), high, low)
        if signal == "NEUTRAL" or pd.isna(row["actual_5d"]):
            continue
        is_correct = (signal == "POSITIVE" and row["actual_5d"] > 0) or (
            signal == "NEGATIVE" and row["actual_5d"] < 0
        )
        rows.append(
            {
                "ticker": row["ticker"],
                "call_date": row["call_date"].date().isoformat(),
                "quarter": row["call_date"].to_period("Q").strftime("%YQ%q"),
                "ew_risk_score": row["ew_risk_score"],
                "threshold_low": round(low, 4),
                "threshold_high": round(high, 4),
                "signal": signal,
                "actual_5d": row["actual_5d"],
                "correct": bool(is_correct),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        logger.warning("Walk-forward produced no directional evaluations.")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        return result_df, float("nan")

    quarter_acc = (
        result_df.groupby("quarter")["correct"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_directional", "sum": "n_correct"})
    )
    walkforward_acc = float(result_df["correct"].mean())

    original_directional = df[df["signal"] != "NEUTRAL"]["correct_5d"].dropna()
    original_full_acc = float(original_directional.mean()) if not original_directional.empty else float("nan")

    original_2025 = df[(df["call_date"] >= pd.Timestamp("2025-01-01")) & (df["signal"] != "NEUTRAL")][
        "correct_5d"
    ].dropna()
    original_2025_acc = float(original_2025.mean()) if not original_2025.empty else float("nan")

    print("\n=== Walk-Forward Accuracy by Quarter (2025+) ===")
    print(quarter_acc.to_string(index=False))
    print("\n=== Walk-Forward Overall ===")
    print(f"Walk-forward directional accuracy: {walkforward_acc:.4f} (n={len(result_df)})")
    print(f"Original full-corpus directional accuracy: {original_full_acc:.4f}")
    print(f"Original 2025+ directional accuracy (static full-corpus thresholds): {original_2025_acc:.4f}")
    print("Note: test-set n is intentionally small in this walk-forward setup.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.success(f"Walk-forward results saved -> {out_path}")
    return result_df, walkforward_acc


if __name__ == "__main__":
    run_walkforward_backtest()
