"""
engine.py
Backtest the EW_Risk_Score signal against post-earnings price returns.

For each scored file:
  - Extract filing_date → next trading day = event day
  - Fetch OHLCV with yfinance for event day + buffer
  - Compute actual 1d / 3d / 5d returns (close-to-close)
  - Assign signal using corpus-wide percentile thresholds (config/settings.py):
      top PERCENTILE_HIGH %  → NEGATIVE (predict stock falls)
      bottom PERCENTILE_LOW % → POSITIVE (predict stock rises)
      middle                  → NEUTRAL (excluded from directional accuracy)
  - Compare prediction vs actual return direction
  - Save full row to data/backtest_results.csv

Usage
-----
python -m src.backtest.engine [--scores-dir data/scores] [--out data/backtest_results.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))
from config.settings import PERCENTILE_HIGH, PERCENTILE_LOW

# Return windows in trading days
RETURN_WINDOWS = [1, 3, 5]

# yfinance courtesy delay between distinct ticker fetches
YF_DELAY = 0.3


# ---------------------------------------------------------------------------
# Price fetching helpers
# ---------------------------------------------------------------------------

def _next_trading_day(event_date: date, prices: pd.DataFrame) -> Optional[date]:
    """Return the first date >= event_date that appears in prices.index."""
    idx = pd.DatetimeIndex(prices.index).normalize()
    event_dt = pd.Timestamp(event_date)
    future = idx[idx >= event_dt]
    if future.empty:
        return None
    return future[0].date()


def _fetch_prices(ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV for ticker between start and end.
    Returns DataFrame indexed by date, or None on failure.
    """
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception as exc:
        logger.debug(f"yfinance error for {ticker}: {exc}")
        return None


def _fetch_ticker_prices(ticker: str, call_dates: list[date]) -> Optional[pd.DataFrame]:
    """
    Fetch a single wide price window covering all call dates for ticker.
    This avoids the broken per-call caching that missed dates.
    """
    if not call_dates:
        return None
    start = min(call_dates) - timedelta(days=10)
    end   = max(call_dates) + timedelta(days=25)
    return _fetch_prices(ticker, start, end)


def _compute_returns(
    prices: pd.DataFrame,
    event_date: date,
) -> dict[str, Optional[float]]:
    """
    Compute 1d / 3d / 5d log returns starting from event_date open.

    We use Close-to-Close from the day before the event to day T+N
    (or Open[event] to Close[event+N]) — here we use adjusted close
    prices: return = Close[T+N] / Close[T-1] - 1.
    """
    results: dict[str, Optional[float]] = {
        "actual_1d": None,
        "actual_3d": None,
        "actual_5d": None,
        "event_date": None,
    }

    if prices is None or prices.empty:
        return results

    # Normalise index to date
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    event_day = _next_trading_day(event_date, prices)
    if event_day is None:
        return results

    results["event_date"] = event_day.isoformat()

    # Locate event day position in the index
    idx = prices.index
    event_pos_arr = (idx == pd.Timestamp(event_day)).nonzero()[0]
    if len(event_pos_arr) == 0:
        return results
    event_pos = int(event_pos_arr[0])

    # Reference: close of the day before event (or open if event_pos == 0)
    if event_pos == 0:
        ref_price = float(prices["Open"].iloc[0])
    else:
        ref_price = float(prices["Close"].iloc[event_pos - 1])

    if ref_price <= 0 or pd.isna(ref_price):
        return results

    close = prices["Close"]
    for window in RETURN_WINDOWS:
        target_pos = event_pos + window - 1
        if target_pos < len(close):
            target_price = float(close.iloc[target_pos])
            if not pd.isna(target_price) and target_price > 0:
                ret = (target_price / ref_price) - 1.0
                results[f"actual_{window}d"] = round(ret, 6)

    return results


# ---------------------------------------------------------------------------
# Signal logic — percentile-based
# ---------------------------------------------------------------------------

def _assign_signals(
    scores: list[float],
    p_high: int = PERCENTILE_HIGH,
    p_low: int = PERCENTILE_LOW,
) -> list[str]:
    """
    Assign NEGATIVE / POSITIVE / NEUTRAL based on corpus-wide percentiles.
    Top p_high% → NEGATIVE, bottom p_low% → POSITIVE, rest → NEUTRAL.
    """
    arr = np.array(scores, dtype=float)
    high_thresh = float(np.percentile(arr, p_high))
    low_thresh  = float(np.percentile(arr, p_low))
    logger.info(
        f"Signal thresholds: P{p_high}={high_thresh:.1f} (NEGATIVE), "
        f"P{p_low}={low_thresh:.1f} (POSITIVE)"
    )
    result = []
    for s in scores:
        if s >= high_thresh:
            result.append("NEGATIVE")
        elif s <= low_thresh:
            result.append("POSITIVE")
        else:
            result.append("NEUTRAL")
    return result


def _correct_direction(signal: str, actual_return: Optional[float]) -> Optional[bool]:
    """True if signal direction matches actual return sign. None if ambiguous."""
    if signal == "NEUTRAL" or actual_return is None:
        return None
    if signal == "POSITIVE":
        return actual_return > 0
    if signal == "NEGATIVE":
        return actual_return < 0
    return None


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    scores_dir: str | Path = _ROOT / "data" / "scores",
    out_path: str | Path = _ROOT / "data" / "backtest_results.csv",
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Run the backtest over all score JSON files in scores_dir.

    Signal assignment uses corpus-wide percentile thresholds (P80/P20 by default)
    configured in config/settings.py — not fixed absolute thresholds.

    Parameters
    ----------
    scores_dir  : directory containing *_score.json files
    out_path    : path to write the results CSV
    sector_map  : {ticker: sector} dict; loaded from universe if None

    Returns
    -------
    pd.DataFrame of backtest results
    """
    scores_dir = Path(scores_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if sector_map is None:
        from src.backtest.universe import ticker_sector_map
        sector_map = ticker_sector_map()

    score_files = sorted(scores_dir.glob("*_score.json"))
    if not score_files:
        logger.error(f"No score files found in {scores_dir}")
        return pd.DataFrame()

    logger.info(f"Running backtest on {len(score_files)} score files")

    # ── Pass 1: load all score data and group call dates by ticker ────────────
    raw_rows: list[dict] = []
    ticker_dates: dict[str, list[date]] = {}

    for fp in score_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not read {fp.name}: {exc}")
            continue

        ticker = data.get("ticker", "")
        call_date_str = data.get("date", "")
        if not ticker or not call_date_str:
            continue
        try:
            call_date = date.fromisoformat(call_date_str)
        except ValueError:
            logger.warning(f"Bad date in {fp.name}: {call_date_str!r}")
            continue

        raw_rows.append({
            "ticker": ticker,
            "call_date": call_date_str,
            "call_date_obj": call_date,
            "ew_risk_score": data.get("EW_Risk_Score", 0.0),
            "hedge_density": data.get("hedging", {}).get("hedging_density", 0.0),
            "neg_sentiment": data.get("sentiment", {}).get("overall_negative_ratio", 0.0),
            "backward_ratio": data.get("vocab", {}).get("backward_ratio", 0.0),
            "sentiment_trajectory": data.get("sentiment", {}).get("sentiment_trajectory", 0.0),
            "sections_found": data.get("sections_found", False),
            "source": data.get("source", "edgar"),
            "sector": sector_map.get(ticker, "Unknown"),
        })
        ticker_dates.setdefault(ticker, []).append(call_date)

    if not raw_rows:
        logger.error("No rows generated — check score files and price data")
        return pd.DataFrame()

    # ── Pass 2: fetch prices per ticker with one wide window ─────────────────
    logger.info(f"Fetching prices for {len(ticker_dates)} tickers...")
    price_cache: dict[str, Optional[pd.DataFrame]] = {}
    for ticker, dates in tqdm(ticker_dates.items(), desc="Prices", unit="ticker"):
        time.sleep(YF_DELAY)
        price_cache[ticker] = _fetch_ticker_prices(ticker, dates)

    # ── Pass 3: compute returns for each row ──────────────────────────────────
    rows: list[dict] = []
    for r in raw_rows:
        ticker = r["ticker"]
        call_date = r["call_date_obj"]
        prices = price_cache.get(ticker)
        returns = _compute_returns(prices, call_date)
        rows.append({
            "ticker": ticker,
            "sector": r["sector"],
            "source": r["source"],
            "sections_found": r["sections_found"],
            "call_date": r["call_date"],
            "event_date": returns.get("event_date"),
            "ew_risk_score": r["ew_risk_score"],
            "hedge_density": round(r["hedge_density"], 4),
            "neg_sentiment": round(r["neg_sentiment"], 4),
            "backward_ratio": round(r["backward_ratio"], 4),
            "sentiment_trajectory": round(r["sentiment_trajectory"], 4),
            "actual_1d": returns.get("actual_1d"),
            "actual_3d": returns.get("actual_3d"),
            "actual_5d": returns.get("actual_5d"),
        })

    df_results = pd.DataFrame(rows)

    # ── Pass 4: assign percentile-based signals across full corpus ────────────
    signals = _assign_signals(df_results["ew_risk_score"].tolist())
    df_results["signal"] = signals

    for w in ["1d", "3d", "5d"]:
        df_results[f"correct_{w}"] = df_results.apply(
            lambda row, w=w: _correct_direction(row["signal"], row.get(f"actual_{w}")),
            axis=1,
        )

    df_results.to_csv(out_path, index=False)
    logger.success(f"Backtest results saved → {out_path}  ({len(df_results)} rows)")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EW_Risk_Score backtest")
    parser.add_argument("--scores-dir", default="data/scores")
    parser.add_argument("--out", default="data/backtest_results.csv")
    args = parser.parse_args()
    run_backtest(scores_dir=args.scores_dir, out_path=args.out)
