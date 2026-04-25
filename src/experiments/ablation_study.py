"""
Ablation study for EarningsEcho signal weight choices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))
from config.settings import PERCENTILE_HIGH, PERCENTILE_LOW
from src.nlp.composite_score import normalize_components


def _load_base_frame() -> pd.DataFrame:
    """Load normalized components from score JSON and target from backtest CSV."""
    backtest = pd.read_csv(_ROOT / "data" / "backtest_results.csv")
    backtest["call_date"] = pd.to_datetime(backtest["call_date"])
    backtest["target"] = (backtest["actual_5d"] > 0).astype(int)

    rows: list[dict] = []
    for score_file in sorted((_ROOT / "data" / "scores").glob("*_score.json")):
        payload = json.loads(score_file.read_text(encoding="utf-8"))
        sentiment = payload.get("sentiment", {})
        norm = normalize_components(
            hedging_density=float(payload.get("hedging", {}).get("hedging_density", 0.0)),
            negative_ratio=float(sentiment.get("overall_negative_ratio", 0.0)),
            backward_ratio=float(payload.get("vocab", {}).get("backward_ratio", 0.0)),
            sentiment_trajectory=float(sentiment.get("sentiment_trajectory", 0.0)),
        )
        rows.append(
            {
                "ticker": payload["ticker"],
                "call_date": pd.to_datetime(payload["date"]),
                "hedging_norm": float(norm["hedging_norm"]),
                "negative_sentiment_norm": float(norm["negative_sentiment_norm"]),
                "backward_ratio_norm": float(norm["backward_ratio_norm"]),
                "sentiment_trajectory_norm": float(norm["sentiment_trajectory_norm"] or 0.0),
            }
        )

    components = pd.DataFrame(rows)
    merged = backtest.merge(components, on=["ticker", "call_date"], how="left")
    merged[[
        "hedging_norm",
        "negative_sentiment_norm",
        "backward_ratio_norm",
        "sentiment_trajectory_norm",
    ]] = merged[[
        "hedging_norm",
        "negative_sentiment_norm",
        "backward_ratio_norm",
        "sentiment_trajectory_norm",
    ]].fillna(0.0)
    return merged


def _directional_accuracy(df: pd.DataFrame, score_col: str) -> tuple[float, int, int]:
    """Compute P80/P20 directional accuracy for a synthetic score column."""
    high = float(np.percentile(df[score_col], PERCENTILE_HIGH))
    low = float(np.percentile(df[score_col], PERCENTILE_LOW))
    signal = np.where(df[score_col] >= high, "NEGATIVE", np.where(df[score_col] <= low, "POSITIVE", "NEUTRAL"))
    directional = df.assign(signal=signal).query("signal != 'NEUTRAL'").copy()
    if directional.empty:
        return float("nan"), 0, 0
    pred_up = (directional["signal"] == "POSITIVE").astype(int)
    correct = (pred_up == directional["target"]).astype(int)
    return float(correct.mean()), int(correct.sum()), int(len(correct))


def run_ablation_study(out_path: str | Path = _ROOT / "data" / "ablation_results.csv") -> pd.DataFrame:
    """Run systematic weight ablations and rank by directional accuracy."""
    df = _load_base_frame()
    combos = [
        ("original_40_35_25", (0.40, 0.35, 0.25, 0.0)),
        ("equal_33_33_33", (0.33, 0.33, 0.33, 0.0)),
        ("hedging_only", (1.0, 0.0, 0.0, 0.0)),
        ("sentiment_only", (0.0, 1.0, 0.0, 0.0)),
        ("vocab_only", (0.0, 0.0, 1.0, 0.0)),
        ("hedging_plus_sentiment", (0.50, 0.50, 0.0, 0.0)),
        ("no_hedging", (0.0, 0.50, 0.50, 0.0)),
        ("four_signal_with_trajectory", (0.35, 0.30, 0.20, 0.15)),
    ]

    rows: list[dict] = []
    for name, (w_h, w_s, w_v, w_t) in combos:
        score_col = f"score_{name}"
        df[score_col] = 100.0 * (
            w_h * df["hedging_norm"]
            + w_s * df["negative_sentiment_norm"]
            + w_v * df["backward_ratio_norm"]
            + w_t * df["sentiment_trajectory_norm"]
        )
        acc, n_correct, n_eval = _directional_accuracy(df, score_col)
        rows.append(
            {
                "combination": name,
                "w_hedging": w_h,
                "w_sentiment": w_s,
                "w_vocab": w_v,
                "w_sentiment_trajectory": w_t,
                "directional_accuracy": round(acc, 4),
                "n_correct": n_correct,
                "n_directional": n_eval,
            }
        )

    results = pd.DataFrame(rows).sort_values("directional_accuracy", ascending=False)
    print("\n=== Ablation Study (Ranked by Directional Accuracy) ===")
    print(results.to_string(index=False))
    print("\nNote: 4th-signal trajectory variant is included as tested-and-rejected documentation.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.success(f"Ablation results saved -> {out_path}")
    return results


if __name__ == "__main__":
    run_ablation_study()
