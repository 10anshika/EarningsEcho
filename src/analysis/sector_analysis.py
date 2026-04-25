"""
Sector-level performance analysis and visualizations for EarningsEcho.

Outputs
-------
- data/sector_analysis_results.csv
- data/plots/sector_accuracy_heatmap.png
- data/plots/hedge_density_by_sector_boxplot.png
- data/plots/signal_accuracy_by_sector.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats

WINDOWS = ("1d", "3d", "5d")
ALPHA = 0.05


def _minimum_n_for_significance(
    observed_accuracy: float,
    baseline: float = 0.5,
    alpha: float = ALPHA,
    max_n: int = 10000,
) -> int | None:
    """Return smallest n with one-tailed binomial p <= alpha."""
    if np.isnan(observed_accuracy) or observed_accuracy <= baseline:
        return None

    for n in range(5, max_n + 1):
        k = int(round(observed_accuracy * n))
        p_val = scipy_stats.binomtest(k, n, p=baseline, alternative="greater").pvalue
        if p_val <= alpha:
            return n
    return None


def _directional_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build sector x window directional accuracy table."""
    rows: list[dict] = []
    directional = df[df["signal"] != "NEUTRAL"].copy()
    for sector, grp in directional.groupby("sector"):
        for w in WINDOWS:
            col = f"correct_{w}"
            valid = grp[col].dropna()
            if valid.empty:
                continue
            acc = float(valid.mean())
            n = int(len(valid))
            rows.append(
                {
                    "metric_group": "directional_accuracy_by_sector_window",
                    "sector": sector,
                    "window": w,
                    "signal_scope": "all_directional",
                    "accuracy": round(acc, 4),
                    "n": n,
                    "n_correct": int(valid.sum()),
                    "p_value_vs_50pct": round(
                        float(scipy_stats.binomtest(int(valid.sum()), n, p=0.5, alternative="greater").pvalue),
                        6,
                    ),
                    "min_n_for_significance": _minimum_n_for_significance(acc),
                }
            )
    return pd.DataFrame(rows)


def _hedge_density_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hedge-density distribution summary and global cross-sector tests."""
    rows: list[dict] = []
    for sector, grp in df.groupby("sector"):
        density = grp["hedge_density"].dropna()
        rows.append(
            {
                "metric_group": "hedge_density_distribution",
                "sector": sector,
                "window": "all",
                "signal_scope": "all_rows",
                "accuracy": np.nan,
                "n": int(len(density)),
                "n_correct": np.nan,
                "p_value_vs_50pct": np.nan,
                "min_n_for_significance": np.nan,
                "hedge_density_mean": round(float(density.mean()), 4),
                "hedge_density_median": round(float(density.median()), 4),
                "hedge_density_std": round(float(density.std(ddof=1)), 4) if len(density) > 1 else 0.0,
            }
        )

    sector_samples = [g["hedge_density"].dropna().to_numpy() for _, g in df.groupby("sector")]
    sector_samples = [x for x in sector_samples if len(x) >= 2]
    if len(sector_samples) >= 2:
        anova = scipy_stats.f_oneway(*sector_samples)
        kruskal = scipy_stats.kruskal(*sector_samples)
        rows.append(
            {
                "metric_group": "hedge_density_global_test",
                "sector": "ALL",
                "window": "all",
                "signal_scope": "all_rows",
                "accuracy": np.nan,
                "n": int(df["hedge_density"].notna().sum()),
                "n_correct": np.nan,
                "p_value_vs_50pct": np.nan,
                "min_n_for_significance": np.nan,
                "hedge_density_mean": np.nan,
                "hedge_density_median": np.nan,
                "hedge_density_std": np.nan,
                "anova_p_value": round(float(anova.pvalue), 6),
                "kruskal_p_value": round(float(kruskal.pvalue), 6),
            }
        )
    return pd.DataFrame(rows)


def _signal_split_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute NEGATIVE vs POSITIVE directional accuracy by sector and window."""
    rows: list[dict] = []
    for sector, grp in df.groupby("sector"):
        for signal_value in ("NEGATIVE", "POSITIVE"):
            signal_grp = grp[grp["signal"] == signal_value]
            for w in WINDOWS:
                col = f"correct_{w}"
                valid = signal_grp[col].dropna()
                if valid.empty:
                    continue
                acc = float(valid.mean())
                n = int(len(valid))
                rows.append(
                    {
                        "metric_group": "signal_split_accuracy_by_sector_window",
                        "sector": sector,
                        "window": w,
                        "signal_scope": signal_value.lower(),
                        "accuracy": round(acc, 4),
                        "n": n,
                        "n_correct": int(valid.sum()),
                        "p_value_vs_50pct": round(
                            float(scipy_stats.binomtest(int(valid.sum()), n, p=0.5, alternative="greater").pvalue),
                            6,
                        ),
                        "min_n_for_significance": _minimum_n_for_significance(acc),
                    }
                )
    return pd.DataFrame(rows)


def _plot_accuracy_heatmap(acc_df: pd.DataFrame, plots_dir: Path) -> None:
    """Save sector-window accuracy heatmap."""
    pivot = acc_df.pivot(index="sector", columns="window", values="accuracy")
    pivot = pivot.reindex(columns=list(WINDOWS))

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_title("Directional Accuracy by Sector and Window")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    fig.savefig(plots_dir / "sector_accuracy_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_hedge_density_boxplot(df: pd.DataFrame, plots_dir: Path) -> None:
    """Save hedge-density by sector boxplot."""
    sectors = sorted(df["sector"].dropna().unique().tolist())
    data = [df[df["sector"] == s]["hedge_density"].dropna().to_numpy() for s in sectors]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=sectors, patch_artist=True)
    ax.set_title("Hedge Density Distribution by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Hedge Density")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots_dir / "hedge_density_by_sector_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_signal_split(split_df: pd.DataFrame, plots_dir: Path) -> None:
    """Save NEGATIVE vs POSITIVE sector accuracy chart on 5d window."""
    sub = split_df[split_df["window"] == "5d"].copy()
    if sub.empty:
        return
    pivot = sub.pivot(index="sector", columns="signal_scope", values="accuracy").sort_index()

    x = np.arange(len(pivot.index))
    width = 0.35
    neg = pivot.get("negative", pd.Series(index=pivot.index, dtype=float)).to_numpy()
    pos = pivot.get("positive", pd.Series(index=pivot.index, dtype=float)).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, neg, width=width, label="NEGATIVE", color="#d62728")
    ax.bar(x + width / 2, pos, width=width, label="POSITIVE", color="#2ca02c")
    ax.set_title("5d Accuracy Split: NEGATIVE vs POSITIVE by Sector")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "signal_accuracy_by_sector.png", dpi=180)
    plt.close(fig)


def run_sector_analysis(
    csv_path: str | Path = Path("data/backtest_results.csv"),
    out_path: str | Path = Path("data/sector_analysis_results.csv"),
) -> pd.DataFrame:
    """Run sector analysis, produce tables + plots, and save output CSV."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    plots_dir = Path("data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Backtest CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    directional_df = _directional_accuracy_table(df)
    hedge_df = _hedge_density_stats(df)
    split_df = _signal_split_accuracy(df)
    results = pd.concat([directional_df, hedge_df, split_df], ignore_index=True, sort=False)

    _plot_accuracy_heatmap(directional_df, plots_dir)
    _plot_hedge_density_boxplot(df, plots_dir)
    _plot_signal_split(split_df, plots_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.success(f"Sector analysis results saved -> {out_path}")

    print("\n=== Sector Directional Accuracy (all directional signals) ===")
    print(directional_df.sort_values(["window", "accuracy"], ascending=[True, False]).to_string(index=False))
    print("\n=== Hedge Density Cross-Sector Test ===")
    print(hedge_df[hedge_df["metric_group"] == "hedge_density_global_test"].to_string(index=False))
    print("\n=== NEGATIVE vs POSITIVE Accuracy Split (5d) ===")
    print(split_df[split_df["window"] == "5d"].sort_values("accuracy", ascending=False).to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector-level analysis for EarningsEcho")
    parser.add_argument("--csv", default="data/backtest_results.csv")
    parser.add_argument("--out", default="data/sector_analysis_results.csv")
    args = parser.parse_args()
    run_sector_analysis(csv_path=args.csv, out_path=args.out)
