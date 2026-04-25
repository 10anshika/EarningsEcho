"""
Compare supervised ML classifiers against the rule-based EarningsEcho signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))
from config.settings import PERCENTILE_HIGH, PERCENTILE_LOW

FEATURES = ["hedging_norm", "negative_sentiment_norm", "backward_ratio_norm"]


def _load_feature_frame() -> pd.DataFrame:
    """Load backtest rows and reconstruct feature columns from score JSON files."""
    backtest_path = _ROOT / "data" / "backtest_results.csv"
    scores_dir = _ROOT / "data" / "scores"

    df = pd.read_csv(backtest_path)
    df["call_date"] = pd.to_datetime(df["call_date"])
    df["target"] = (df["actual_5d"] > 0).astype(int)

    feature_rows: list[dict] = []
    for score_file in sorted(scores_dir.glob("*_score.json")):
        payload = json.loads(score_file.read_text(encoding="utf-8"))
        feature_rows.append(
            {
                "ticker": payload["ticker"],
                "call_date": pd.to_datetime(payload["date"]),
                "hedging_norm": float(payload.get("hedging_norm", 0.0)),
                "negative_sentiment_norm": float(payload.get("negative_sentiment_norm", 0.0)),
                "backward_ratio_norm": float(payload.get("backward_ratio_norm", 0.0)),
            }
        )

    feat_df = pd.DataFrame(feature_rows)
    merged = df.merge(feat_df, on=["ticker", "call_date"], how="left")
    missing = merged[FEATURES].isna().any(axis=1).sum()
    if missing:
        logger.warning(f"{missing} rows missing reconstructed feature values; filling with 0.0")
        merged[FEATURES] = merged[FEATURES].fillna(0.0)
    return merged.sort_values("call_date").reset_index(drop=True)


def _metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute standard binary classification metrics for one model."""
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "n_eval": int(len(y_true)),
        "coverage": 1.0,
    }


def _evaluate_rule_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Evaluate rule-based P80/P20 baseline using train-period thresholds only."""
    high_thresh = float(np.percentile(train_df["ew_risk_score"].to_numpy(dtype=float), PERCENTILE_HIGH))
    low_thresh = float(np.percentile(train_df["ew_risk_score"].to_numpy(dtype=float), PERCENTILE_LOW))

    signals = np.where(
        test_df["ew_risk_score"] >= high_thresh,
        "NEGATIVE",
        np.where(test_df["ew_risk_score"] <= low_thresh, "POSITIVE", "NEUTRAL"),
    )
    eval_df = test_df.copy()
    eval_df["signal"] = signals
    eval_df = eval_df[eval_df["signal"] != "NEUTRAL"].copy()
    if eval_df.empty:
        return {
            "model": "RuleBaseline_P80P20",
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "n_eval": 0,
            "coverage": 0.0,
        }

    y_true = eval_df["target"]
    y_pred = (eval_df["signal"] == "POSITIVE").astype(int)
    return {
        "model": "RuleBaseline_P80P20",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred)), 4),
        "n_eval": int(len(eval_df)),
        "coverage": round(float(len(eval_df) / len(test_df)), 4),
    }


def run_ml_comparison(out_path: str | Path = _ROOT / "data" / "ml_comparison_results.csv") -> pd.DataFrame:
    """Run chronological ML model comparison and persist metrics table."""
    df = _load_feature_frame()
    train_df = df[df["call_date"] < pd.Timestamp("2025-01-01")].copy()
    test_df = df[df["call_date"] >= pd.Timestamp("2025-01-01")].copy()

    logger.info(f"Chronological split: train={len(train_df)} rows, test={len(test_df)} rows")
    X_train, y_train = train_df[FEATURES], train_df["target"]
    X_test, y_test = test_df[FEATURES], test_df["target"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    rows: list[dict] = []
    importance_rows: list[dict] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append(_metric_row(name, y_test, y_pred, y_prob))

        if hasattr(model, "feature_importances_"):
            for feature_name, importance in zip(FEATURES, model.feature_importances_):
                importance_rows.append(
                    {
                        "model": name,
                        "metric": f"feature_importance_{feature_name}",
                        "value": round(float(importance), 4),
                    }
                )

    rows.append(_evaluate_rule_baseline(train_df, test_df))
    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False, na_position="last")

    print("\n=== ML vs Rule-Based Comparison (Chronological Split) ===")
    print(results.to_string(index=False))

    if importance_rows:
        importances = pd.DataFrame(importance_rows)
        print("\n=== Tree Model Feature Importances ===")
        print(importances.pivot(index="model", columns="metric", values="value").to_string())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.success(f"ML comparison results saved -> {out_path}")
    return results


if __name__ == "__main__":
    run_ml_comparison()
