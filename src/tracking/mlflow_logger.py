"""
mlflow_logger.py
Log EarningsEcho pipeline runs to MLflow.

Public API
----------
log_run(score_data, score_path, backtest_csv) -> str   # returns run_id
get_recent_runs(n=10) -> list[dict]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# MLflow experiment config
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = "EarningsEcho"
TRACKING_URI     = "http://localhost:5000"
FINBERT_MODEL    = "ProsusAI/finbert"

_ROOT = Path(__file__).parents[2]
BACKTEST_CSV = _ROOT / "data" / "backtest_results.csv"


def _setup_mlflow() -> None:
    """Point MLflow at the local tracking server; create experiment if absent."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_run(
    score_data: dict,
    score_path: Optional[str | Path] = None,
    backtest_csv: Optional[str | Path] = None,
) -> Optional[str]:
    """
    Log a single pipeline run to MLflow.

    Parameters
    ----------
    score_data   : result dict from nlp_pipeline.analyze()
    score_path   : path to the saved score JSON (logged as artifact)
    backtest_csv : path to backtest CSV; used to look up actual_5d return

    Returns
    -------
    MLflow run_id string, or None if logging fails (non-fatal).
    """
    ticker    = score_data.get("ticker", "UNKNOWN")
    call_date = score_data.get("date", "unknown")
    run_name  = f"{ticker}_{call_date}"

    try:
        _setup_mlflow()
    except Exception as exc:
        logger.warning(f"MLflow setup failed (server not running?): {exc}")
        return None

    _captured: dict = {}  # holds run_id once the run body completes
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # ── Parameters ────────────────────────────────────────────────
            mlflow.log_params({
                "ticker":         ticker,
                "filing_date":    call_date,
                "whisper_model":  "none/text-only",
                "finbert_model":  FINBERT_MODEL,
                "source":         score_data.get("source", "edgar"),
                "sections_found": str(score_data.get("sections_found", False)),
                "risk_class":     score_data.get("risk_class", ""),
            })

            # ── Metrics ───────────────────────────────────────────────────
            hedge_raw = score_data.get("hedging", {})
            sent_raw  = score_data.get("sentiment", {})
            vocab_raw = score_data.get("vocab", {})

            metrics: dict[str, float] = {
                "ew_risk_score":   float(score_data.get("EW_Risk_Score", 0)),
                "hedge_density":   float(hedge_raw.get("hedging_density", 0)),
                "neg_sentiment":   float(sent_raw.get("overall_negative_ratio", 0)),
                "backward_ratio":  float(vocab_raw.get("backward_ratio", 0)),
                "hedging_norm":    float(score_data.get("hedging_norm", 0)),
                "neg_sent_norm":   float(score_data.get("negative_sentiment_norm", 0)),
                "backward_norm":   float(score_data.get("backward_ratio_norm", 0)),
                "sent_trajectory": float(sent_raw.get("sentiment_trajectory", 0)),
                "word_count":      float(score_data.get("word_count", 0)),
            }

            # Actual return from backtest CSV if available
            csv_path = Path(backtest_csv) if backtest_csv else BACKTEST_CSV
            actual_return = _lookup_actual_return(ticker, call_date, csv_path)
            if actual_return is not None:
                metrics["actual_5d_return"] = actual_return

            mlflow.log_metrics(metrics)

            # ── Artifact — score JSON ──────────────────────────────────────
            if score_path is not None:
                sp = Path(score_path)
                if sp.exists():
                    mlflow.log_artifact(str(sp), artifact_path="scores")

            # Capture run_id inside the with-block so we keep it even if
            # context-manager __exit__ raises (MLflow 3.10 writes a Unicode
            # emoji to stdout which fails on Windows cp1252).
            _captured["run_id"] = run.info.run_id

    except UnicodeEncodeError:
        # MLflow's _log_url tries to print a 🏃 emoji on exit — the run was
        # logged successfully; the UnicodeEncodeError is cosmetic only.
        pass
    except Exception as exc:
        logger.warning(f"MLflow log_run failed: {exc}")
        return None

    run_id = _captured.get("run_id")
    if run_id:
        logger.success(f"MLflow run logged: {run_name}  (run_id={run_id[:8]}...)")
    return run_id


def get_recent_runs(n: int = 10) -> list[dict]:
    """
    Return the last *n* runs from the EarningsEcho MLflow experiment.

    Returns
    -------
    List of dicts with keys: run_id, ticker, filing_date, ew_risk_score,
    actual_5d_return, risk_class, status, start_time.
    Empty list if server is unreachable.
    """
    try:
        _setup_mlflow()
        client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return []

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )

        result = []
        for r in runs:
            params  = r.data.params
            metrics = r.data.metrics
            result.append({
                "run_id":           r.info.run_id[:8],
                "ticker":           params.get("ticker", ""),
                "filing_date":      params.get("filing_date", ""),
                "ew_risk_score":    round(metrics.get("ew_risk_score", float("nan")), 2),
                "actual_5d_return": round(metrics.get("actual_5d_return", float("nan")), 4)
                                    if "actual_5d_return" in metrics else None,
                "risk_class":       params.get("risk_class", ""),
                "source":           params.get("source", ""),
                "start_time":       pd.Timestamp(r.info.start_time, unit="ms")
                                    .strftime("%Y-%m-%d %H:%M"),
            })
        return result

    except Exception as exc:
        logger.debug(f"MLflow get_recent_runs failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup_actual_return(
    ticker: str, call_date: str, csv_path: Path
) -> Optional[float]:
    """Return the actual_5d return for (ticker, call_date) from the backtest CSV, or None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        row = df[(df["ticker"] == ticker) & (df["call_date"] == call_date)]
        if row.empty:
            return None
        val = row["actual_5d"].iloc[0]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None
