"""
stats.py
Compute and print a clean summary of backtest results.

Usage
-----
python -m src.backtest.stats [--csv data/backtest_results.csv] [--window 5d]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from loguru import logger


ANNUALIZATION_FACTOR = np.sqrt(252)   # daily returns → annualised Sharpe


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _directional_accuracy(df: pd.DataFrame, col: str) -> tuple[float, int]:
    """
    Return (accuracy, n_directional) for non-NEUTRAL, non-NaN rows.
    col should be 'correct_1d', 'correct_3d', or 'correct_5d'.
    """
    directional = df[df["signal"] != "NEUTRAL"][col].dropna()
    if directional.empty:
        return float("nan"), 0
    return float(directional.mean()), len(directional)


def _binomial_pvalue(n_correct: int, n_total: int) -> float:
    """One-tailed binomial test: H0 = 50% accuracy."""
    if n_total == 0:
        return float("nan")
    result = scipy_stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
    return float(result.pvalue)


def _signal_sharpe(df: pd.DataFrame, return_col: str) -> float:
    """
    Compute annualised Sharpe of a strategy that:
      - Goes long when signal == POSITIVE
      - Goes short when signal == NEGATIVE
      - Stays flat when NEUTRAL
    """
    sub = df[df["signal"] != "NEUTRAL"].copy()
    if sub.empty or return_col not in sub.columns:
        return float("nan")

    returns = sub[return_col].dropna()
    if len(returns) < 3:
        return float("nan")

    # Flip returns for SHORT trades
    signed_returns = returns.where(sub.loc[returns.index, "signal"] == "POSITIVE", -returns)
    mu = signed_returns.mean()
    sigma = signed_returns.std()
    if sigma == 0 or pd.isna(sigma):
        return float("nan")
    return float(mu / sigma * ANNUALIZATION_FACTOR)


def _buy_hold_sharpe(df: pd.DataFrame, return_col: str) -> float:
    """Sharpe if you simply bought every stock at every filing date."""
    returns = df[return_col].dropna()
    if len(returns) < 3:
        return float("nan")
    mu, sigma = returns.mean(), returns.std()
    if sigma == 0 or pd.isna(sigma):
        return float("nan")
    return float(mu / sigma * ANNUALIZATION_FACTOR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _source_accuracy(
    df: pd.DataFrame,
    correct_col: str,
) -> dict[str, dict]:
    """Accuracy broken down by source (edgar / motleyfool)."""
    result: dict[str, dict] = {}
    for src, grp in df[df["signal"] != "NEUTRAL"].groupby("source"):
        valid = grp[correct_col].dropna()
        if valid.empty:
            continue
        n = len(valid)
        n_correct = int(valid.sum())
        acc = n_correct / n
        bt = scipy_stats.binomtest(n_correct, n, p=0.5, alternative="greater")
        result[src] = {
            "accuracy": round(acc, 4),
            "n": n,
            "n_correct": n_correct,
            "p_value": round(float(bt.pvalue), 4),
        }
    return result


def _neg_only_accuracy(df: pd.DataFrame, correct_col: str) -> dict:
    """Accuracy for NEGATIVE signals only (the key risk-detection metric)."""
    neg = df[df["signal"] == "NEGATIVE"][correct_col].dropna()
    if neg.empty:
        return {}
    n, n_correct = len(neg), int(neg.sum())
    bt = scipy_stats.binomtest(n_correct, n, p=0.5, alternative="greater")
    return {
        "accuracy": round(n_correct / n, 4),
        "n": n,
        "n_correct": n_correct,
        "p_value": round(float(bt.pvalue), 4),
    }


def compute_stats(
    df: pd.DataFrame,
    primary_window: str = "5d",
) -> dict:
    """
    Compute full backtest statistics from the results DataFrame.

    Parameters
    ----------
    df             : output of engine.run_backtest()
    primary_window : '1d', '3d', or '5d' — used for headline metrics

    Returns
    -------
    dict of computed statistics
    """
    correct_col = f"correct_{primary_window}"
    return_col = f"actual_{primary_window}"

    if correct_col not in df.columns:
        raise ValueError(f"Column '{correct_col}' not in DataFrame")

    # ── Overall accuracy ──────────────────────────────────────────────────
    acc, n_directional = _directional_accuracy(df, correct_col)
    n_correct = int(df[df["signal"] != "NEUTRAL"][correct_col].dropna().sum())
    p_val = _binomial_pvalue(n_correct, n_directional)

    # ── Signal distribution ───────────────────────────────────────────────
    signal_counts = df["signal"].value_counts().to_dict()

    # ── Accuracy by sector ────────────────────────────────────────────────
    sector_stats: dict[str, dict] = {}
    for sector, group in df[df["signal"] != "NEUTRAL"].groupby("sector"):
        valid = group[correct_col].dropna()
        if valid.empty:
            continue
        sector_stats[sector] = {
            "accuracy": round(float(valid.mean()), 4),
            "n": int(len(valid)),
            "n_correct": int(valid.sum()),
        }

    # ── Sharpe ────────────────────────────────────────────────────────────
    signal_sharpe = _signal_sharpe(df, return_col)
    bh_sharpe = _buy_hold_sharpe(df, return_col)

    # ── Per-window accuracy ───────────────────────────────────────────────
    window_stats: dict[str, dict] = {}
    for w in ["1d", "3d", "5d"]:
        a, n = _directional_accuracy(df, f"correct_{w}")
        window_stats[w] = {"accuracy": round(a, 4) if not np.isnan(a) else None, "n": n}

    # ── Source breakdown (EDGAR vs Motley Fool) ───────────────────────────
    source_stats = _source_accuracy(df, correct_col) if "source" in df.columns else {}
    neg_stats = _neg_only_accuracy(df, correct_col)

    return {
        "n_total": len(df),
        "n_directional": n_directional,
        "n_correct": n_correct,
        "overall_accuracy": round(acc, 4) if not np.isnan(acc) else None,
        "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
        "signal_counts": signal_counts,
        "sector_accuracy": sector_stats,
        "window_accuracy": window_stats,
        "signal_sharpe": round(signal_sharpe, 3) if not np.isnan(signal_sharpe) else None,
        "buy_hold_sharpe": round(bh_sharpe, 3) if not np.isnan(bh_sharpe) else None,
        "primary_window": primary_window,
        "source_accuracy": source_stats,
        "negative_signal_accuracy": neg_stats,
    }


def print_summary(stats: dict) -> None:
    """Pretty-print the backtest summary."""
    w = stats["primary_window"]
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  EarningsEcho Backtest Summary  (primary window: {w})")
    print(sep)
    print(f"  Total filings scored:   {stats['n_total']}")
    print(f"  Directional signals:    {stats['n_directional']}")
    print()

    acc = stats["overall_accuracy"]
    pv = stats["p_value"]
    print(f"  Directional Accuracy:   {acc*100:.1f}%   (p={pv:.4f})" if acc else "  Directional Accuracy:  N/A")
    print(f"  Signal Sharpe:          {stats['signal_sharpe']}")
    print(f"  Buy-and-Hold Sharpe:    {stats['buy_hold_sharpe']}")
    print()

    print("  Signal distribution:")
    for sig, cnt in sorted(stats["signal_counts"].items()):
        print(f"    {sig:<10}  {cnt}")
    print()

    print("  Accuracy by window:")
    for wnd, wstats in stats["window_accuracy"].items():
        a = wstats["accuracy"]
        n = wstats["n"]
        print(f"    {wnd}:  {a*100:.1f}%  (n={n})" if a else f"    {wnd}:  N/A  (n={n})")
    print()

    if stats["sector_accuracy"]:
        print("  Accuracy by sector:")
        for sector, ss in sorted(stats["sector_accuracy"].items()):
            print(
                f"    {sector:<14}  {ss['accuracy']*100:.1f}%  "
                f"({ss['n_correct']}/{ss['n']})"
            )

    # ── NEGATIVE signal detail ────────────────────────────────────────────
    neg = stats.get("negative_signal_accuracy", {})
    if neg:
        print()
        print("  NEGATIVE signal (risk-detection) accuracy:")
        print(
            f"    All sources:  {neg['accuracy']*100:.1f}%  "
            f"({neg['n_correct']}/{neg['n']})  p={neg['p_value']:.4f}"
        )

    # ── Source breakdown ──────────────────────────────────────────────────
    src = stats.get("source_accuracy", {})
    if src:
        print()
        print("  Accuracy by transcript source (NEGATIVE signals only via source):")
        for s, ss in sorted(src.items()):
            label = "Motley Fool" if s == "motleyfool" else "EDGAR"
            print(
                f"    {label:<14}  {ss['accuracy']*100:.1f}%  "
                f"({ss['n_correct']}/{ss['n']})  p={ss['p_value']:.4f}"
            )
    print(sep)


def run(
    csv_path: str | Path = Path("data/backtest_results.csv"),
    primary_window: str = "5d",
) -> dict:
    """Load backtest CSV, compute stats, print summary, and return stats dict."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    s = compute_stats(df, primary_window=primary_window)
    print_summary(s)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EarningsEcho backtest statistics")
    parser.add_argument("--csv", default="data/backtest_results.csv")
    parser.add_argument("--window", default="5d", choices=["1d", "3d", "5d"])
    args = parser.parse_args()
    run(csv_path=args.csv, primary_window=args.window)
