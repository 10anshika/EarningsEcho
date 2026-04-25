"""
Run all EarningsEcho experiment modules in a single reproducible command.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.experiments import run_ablation_study, run_ml_comparison, run_walkforward_backtest


def _original_accuracy() -> float:
    """Load original directional accuracy from backtest results."""
    df = pd.read_csv(Path("data") / "backtest_results.csv")
    directional = df[df["signal"] != "NEUTRAL"]["correct_5d"].dropna()
    if directional.empty:
        return float("nan")
    return float(directional.mean())


def main() -> None:
    """Execute all experiment studies and print a final summary."""
    logger.info("Running ML comparison...")
    ml_results = run_ml_comparison()

    logger.info("Running ablation study...")
    run_ablation_study()

    logger.info("Running walk-forward backtest...")
    _, walkforward_acc = run_walkforward_backtest()

    original_acc = _original_accuracy()
    best_ml_row = ml_results[ml_results["model"] != "RuleBaseline_P80P20"].sort_values(
        "accuracy",
        ascending=False,
    ).iloc[0]

    summary = pd.DataFrame(
        [
            {"metric": "Original directional accuracy", "value": round(original_acc, 4)},
            {"metric": "Walk-forward directional accuracy", "value": round(walkforward_acc, 4)},
            {"metric": f"Best ML model accuracy ({best_ml_row['model']})", "value": round(float(best_ml_row["accuracy"]), 4)},
        ]
    )

    print("\n=== Final Experiment Summary ===")
    print(summary.to_string(index=False))
    print("\nArtifacts:")
    print("- data/ml_comparison_results.csv")
    print("- data/ablation_results.csv")
    print("- data/walkforward_results.csv")


if __name__ == "__main__":
    main()
