"""
Statistical power analysis utilities for EarningsEcho directional accuracies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


def cohens_h(p: float, p0: float = 0.50) -> float:
    """
    Compute Cohen's h effect size between two proportions.

    Cohen's h = 2 * (asin(sqrt(p)) - asin(sqrt(p0))).
    This function returns the absolute value.
    """
    p_clipped = float(np.clip(p, 1e-9, 1 - 1e-9))
    p0_clipped = float(np.clip(p0, 1e-9, 1 - 1e-9))
    h = 2.0 * (np.arcsin(np.sqrt(p_clipped)) - np.arcsin(np.sqrt(p0_clipped)))
    return float(abs(h))


def min_n_for_significance(
    p: float,
    p0: float = 0.50,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Estimate minimum sample size n to detect observed effect with target power.

    Uses normal approximation:
        n = ((z_alpha + z_beta) / h)^2
    where h is Cohen's h and z values are from scipy.stats.norm.ppf.
    """
    h = cohens_h(p, p0=p0)
    if h == 0:
        return int(1_000_000_000)

    alpha_tail = alpha / 2 if two_sided else alpha
    z_alpha = float(norm.ppf(1 - alpha_tail))
    z_beta = float(norm.ppf(power))
    n = ((z_alpha + z_beta) / h) ** 2
    return int(np.ceil(n))


def achieved_power(
    p: float,
    n: int,
    p0: float = 0.50,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Compute achieved statistical power for observed accuracy p and sample size n.

    Uses normal-approximation relationship:
        power = Phi(sqrt(n)*h - z_alpha)
    """
    if n <= 0:
        return float("nan")
    h = cohens_h(p, p0=p0)
    alpha_tail = alpha / 2 if two_sided else alpha
    z_alpha = float(norm.ppf(1 - alpha_tail))
    z_value = np.sqrt(n) * h - z_alpha
    return float(norm.cdf(z_value))


def _effect_label(h: float) -> str:
    """Map Cohen's h to standard qualitative bins."""
    if h < 0.20:
        return "negligible"
    if h < 0.50:
        return "small"
    if h < 0.80:
        return "medium"
    return "large"


def _interpret_power(power_value: float) -> str:
    """Human-readable interpretation of achieved power magnitude."""
    if np.isnan(power_value):
        return "insufficient_data"
    if power_value < 0.50:
        return "underpowered"
    if power_value < 0.80:
        return "moderate_power"
    return "well_powered"


def _accuracy_levels() -> list[float]:
    """Return canonical accuracy levels used across power tables."""
    return [0.51, 0.538, 0.545, 0.5545, 0.5755, 0.60, 0.645, 0.714]


def build_effect_size_table() -> pd.DataFrame:
    """Build effect-size table for predefined accuracy levels."""
    rows: list[dict] = []
    for acc in _accuracy_levels():
        h = cohens_h(acc, p0=0.50)
        rows.append(
            {
                "accuracy": acc,
                "cohens_h": round(h, 6),
                "label": _effect_label(h),
            }
        )
    return pd.DataFrame(rows)


def build_min_n_table() -> pd.DataFrame:
    """Build minimum-n table across accuracy levels and power targets."""
    rows: list[dict] = []
    power_targets = [0.50, 0.80, 0.90]
    for acc in _accuracy_levels():
        h = cohens_h(acc, p0=0.50)
        for target_power in power_targets:
            rows.append(
                {
                    "accuracy": acc,
                    "cohens_h": round(h, 6),
                    "target_power": target_power,
                    "min_n": min_n_for_significance(
                        p=acc,
                        p0=0.50,
                        alpha=0.05,
                        power=target_power,
                        two_sided=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_achieved_power_table() -> pd.DataFrame:
    """Build achieved-power table for key EarningsEcho subsets."""
    subsets = [
        ("All signals", 0.538, 106),
        ("EDGAR", 0.565, 228),
        ("Motley Fool NEGATIVE", 0.714, 33),
        ("Walk-forward", 0.5545, 110),
        ("Consumer 5d", 0.645, 31),
        ("Healthcare 5d", 0.565, 23),
        ("Financials 5d", 0.517, 29),
        ("Technology 5d", 0.417, 12),
        ("Energy 5d", 0.364, 11),
        ("Ablation equal_33", 0.5755, 106),
        ("Ablation original_40_35_25", 0.5472, 106),
    ]

    rows: list[dict] = []
    for subset, acc, n in subsets:
        h = cohens_h(acc, p0=0.50)
        pow_val = achieved_power(p=acc, n=n, p0=0.50, alpha=0.05, two_sided=True)
        min_n_80 = min_n_for_significance(
            p=acc,
            p0=0.50,
            alpha=0.05,
            power=0.80,
            two_sided=True,
        )
        rows.append(
            {
                "subset": subset,
                "n": n,
                "accuracy": acc,
                "cohens_h": round(h, 6),
                "achieved_power": round(pow_val, 6),
                "min_n_for_80pct_power": min_n_80,
                "interpretation": _interpret_power(pow_val),
            }
        )
    return pd.DataFrame(rows)


def _build_power_interpretation() -> str:
    """Construct plain-English interpretation string with computed values."""
    acc = 0.538
    n = 106
    h = cohens_h(acc, p0=0.50)
    min_n_80 = min_n_for_significance(p=acc, p0=0.50, alpha=0.05, power=0.80, two_sided=True)
    pow_val = achieved_power(p=acc, n=n, p0=0.50, alpha=0.05, two_sided=True)
    return (
        "Why p=0.31 at n=106 is expected (not a failure):\n"
        f"- The observed edge (53.8% vs 50%) corresponds to a small effect size (Cohen's h={h:.4f}).\n"
        f"- Detecting an effect this small with 80% power needs about n={min_n_80} directional samples.\n"
        f"- With only n={n}, achieved power is about {pow_val:.3f}, so non-significant p-values are likely.\n"
        "- In short, the directionality signal can still be economically meaningful, but the current sample is "
        "statistically underpowered for strong hypothesis-test confidence."
    )


POWER_INTERPRETATION = _build_power_interpretation()


def run_power_analysis(out_dir: Path = Path("data")) -> dict[str, pd.DataFrame]:
    """
    Run power analysis tables, save CSV outputs, print summaries, and return tables.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effect_df = build_effect_size_table()
    min_n_df = build_min_n_table()
    achieved_df = build_achieved_power_table()

    effect_path = out_dir / "power_effect_size_table.csv"
    min_n_path = out_dir / "power_min_n_table.csv"
    achieved_path = out_dir / "power_achieved_table.csv"

    effect_df.to_csv(effect_path, index=False)
    min_n_df.to_csv(min_n_path, index=False)
    achieved_df.to_csv(achieved_path, index=False)

    logger.success(f"Saved effect size table -> {effect_path}")
    logger.success(f"Saved min-n table -> {min_n_path}")
    logger.success(f"Saved achieved-power table -> {achieved_path}")

    print("\n=== Effect Size Table ===")
    print(effect_df.to_string(index=False))
    print("\n=== Minimum N Table ===")
    print(min_n_df.to_string(index=False))
    print("\n=== Achieved Power Table ===")
    print(achieved_df.to_string(index=False))
    print("\n=== Interpretation ===")
    print(POWER_INTERPRETATION)

    return {
        "effect_size": effect_df,
        "min_n": min_n_df,
        "achieved_power": achieved_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EarningsEcho statistical power analysis")
    parser.add_argument("--out-dir", default="data", help="Directory for CSV outputs")
    args = parser.parse_args()
    run_power_analysis(out_dir=Path(args.out_dir))
