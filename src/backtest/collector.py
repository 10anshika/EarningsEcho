"""
collector.py
For each ticker in the universe: fetch transcripts from EDGAR, parse, score.
Skips files already present in data/scores/.

Usage
-----
python -m src.backtest.collector [--tickers AAPL MSFT ...] [--n 6]
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Project root on sys.path so imports resolve regardless of CWD
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest.universe import load_universe
from src.ingestion.edgar_fetcher import fetch_transcripts
from src.ingestion.transcript_parser import parse_transcript
from src.nlp.nlp_pipeline import analyze
from src.nlp.finbert_scorer import load_pipeline

import json

TRANSCRIPTS_DIR = _ROOT / "data" / "transcripts"
SCORES_DIR = _ROOT / "data" / "scores"
INTER_TICKER_DELAY = 1.5   # seconds between tickers (EDGAR courtesy)


def _score_exists(ticker: str, date: str) -> bool:
    """Return True if a score file already exists for this ticker+date."""
    return (SCORES_DIR / f"{ticker}_{date}_score.json").exists()


def _parsed_exists(ticker: str, date: str) -> bool:
    """Return True if a parsed transcript JSON already exists for (ticker, date)."""
    return (TRANSCRIPTS_DIR / f"{ticker}_{date}_parsed.json").exists()


def _raw_exists(ticker: str, date: str) -> bool:
    """Check if any raw json for this ticker+date exists."""
    return bool(list(TRANSCRIPTS_DIR.glob(f"{ticker}_{date}_*.json")))


def collect(
    tickers: list[str] | None = None,
    n_per_ticker: int = 6,
    pipe=None,
) -> dict:
    """
    Run the full fetch → parse → score pipeline for every ticker.

    Parameters
    ----------
    tickers        : list of tickers to process; None = full universe
    n_per_ticker   : number of 8-K filings to fetch per ticker
    pipe           : pre-loaded FinBERT pipeline (loaded once if None)

    Returns
    -------
    dict with keys: processed, skipped, failed, total_scores
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    if tickers:
        tickers_upper = {t.upper() for t in tickers}
        universe = [e for e in universe if e["ticker"] in tickers_upper]

    logger.info(f"Universe: {len(universe)} tickers, {n_per_ticker} filings each")
    logger.info(f"Target: ~{len(universe) * n_per_ticker} scored transcripts")

    # Load FinBERT once for the whole run
    if pipe is None:
        pipe = load_pipeline()

    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_scores": 0}
    failed_tickers: list[str] = []

    outer_bar = tqdm(universe, desc="Tickers", unit="ticker", position=0)

    for entry in outer_bar:
        ticker = entry["ticker"]
        outer_bar.set_postfix(ticker=ticker)

        try:
            # ── Step 1: Fetch raw transcripts from EDGAR ──────────────────
            raw_transcripts = fetch_transcripts(ticker, n=n_per_ticker)
            if not raw_transcripts:
                logger.warning(f"{ticker}: no transcripts found on EDGAR")
                stats["failed"] += 1
                failed_tickers.append(ticker)
                time.sleep(INTER_TICKER_DELAY)
                continue

            # ── Step 2: Save raw, parse, score each transcript ────────────
            ticker_scores = 0
            for t in tqdm(
                raw_transcripts,
                desc=f"  {ticker}",
                unit="filing",
                leave=False,
                position=1,
            ):
                date = t["filed_date"]

                # Skip if score already exists
                if _score_exists(ticker, date):
                    logger.debug(f"{ticker} {date}: score already exists, skipping")
                    stats["skipped"] += 1
                    ticker_scores += 1
                    continue

                # Save raw JSON if not already there
                raw_path = TRANSCRIPTS_DIR / (
                    f"{ticker}_{date}_{t['accession_number'].replace('-','')}.json"
                )
                if not raw_path.exists():
                    raw_path.write_text(
                        json.dumps(t, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                # Parse
                try:
                    parsed = parse_transcript(raw_path)
                except Exception as exc:
                    logger.warning(f"{ticker} {date}: parse failed — {exc}")
                    continue

                # Skip if opening_remarks too short to be meaningful
                if parsed.get("word_count", 0) < 50:
                    logger.debug(f"{ticker} {date}: too short, skipping")
                    continue

                # Save parsed JSON
                parsed_path = TRANSCRIPTS_DIR / f"{ticker}_{date}_parsed.json"
                parsed_path.write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                # Score
                try:
                    result = analyze(parsed_path, pipe=pipe)
                except Exception as exc:
                    logger.warning(f"{ticker} {date}: NLP analysis failed — {exc}")
                    continue

                # Save score
                score_path = SCORES_DIR / f"{ticker}_{date}_score.json"
                score_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )

                ticker_scores += 1
                stats["total_scores"] += 1

            stats["processed"] += 1
            logger.info(f"{ticker}: {ticker_scores} scores total")

        except Exception as exc:
            logger.error(f"{ticker}: unexpected error — {exc}")
            stats["failed"] += 1
            failed_tickers.append(ticker)

        # Courtesy delay between tickers
        time.sleep(INTER_TICKER_DELAY)

    outer_bar.close()

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(
        f"Collection complete — "
        f"processed={stats['processed']} | "
        f"skipped={stats['skipped']} | "
        f"failed={stats['failed']} | "
        f"new_scores={stats['total_scores']}"
    )
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and score earnings transcripts")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Subset of tickers (default: full universe)"
    )
    parser.add_argument("--n", type=int, default=6, help="Filings per ticker")
    args = parser.parse_args()

    collect(tickers=args.tickers, n_per_ticker=args.n)
