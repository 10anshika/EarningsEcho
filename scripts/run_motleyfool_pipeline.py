"""
run_motleyfool_pipeline.py
Fetch Motley Fool transcripts for 10 volatile tickers, parse, score,
and print a summary table showing EW_Risk_Score per filing.

Usage
-----
python scripts/run_motleyfool_pipeline.py [--n 4] [--tickers INTC PFE ...]
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

# Make sure the project root is on sys.path
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from src.ingestion.motleyfool_fetcher import search_transcripts, TRANSCRIPT_CATALOG
from src.ingestion.transcript_parser import parse_transcript
from src.nlp.nlp_pipeline import analyze
from src.nlp.finbert_scorer import load_pipeline

TRANSCRIPTS_DIR = _ROOT / "data" / "transcripts"
SCORES_DIR = _ROOT / "data" / "scores"

VOLATILE_TICKERS = list(TRANSCRIPT_CATALOG.keys())  # 10 tickers


def _score_exists(ticker: str, date: str) -> Path | None:
    p = SCORES_DIR / f"{ticker}_{date}_score.json"
    return p if p.exists() else None


def run(tickers: list[str], n: int, pipe) -> list[dict]:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for ticker in tickers:
        logger.info(f"── {ticker} ────────────────────────────────")
        transcripts = search_transcripts(ticker, n=n)

        if not transcripts:
            logger.warning(f"{ticker}: no transcripts fetched")
            continue

        for t in transcripts:
            date = t["filed_date"]

            # Skip if already scored
            existing = _score_exists(ticker, date)
            if existing:
                logger.debug(f"{ticker} {date}: already scored, loading")
                score_data = json.loads(existing.read_text(encoding="utf-8"))
                rows.append(_row_from_score(score_data))
                continue

            # Save raw JSON
            slug_short = t["accession_number"].replace("-", "_")[:50]
            raw_path = TRANSCRIPTS_DIR / f"{ticker}_{date}_{slug_short}_mf.json"
            if not raw_path.exists():
                raw_path.write_text(
                    json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            # Parse
            try:
                parsed = parse_transcript(raw_path)
            except Exception as exc:
                logger.warning(f"{ticker} {date}: parse failed — {exc}")
                continue

            if parsed.get("word_count", 0) < 50:
                logger.debug(f"{ticker} {date}: too short, skipping")
                continue

            # Save parsed JSON
            parsed_path = TRANSCRIPTS_DIR / f"{ticker}_{date}_parsed.json"
            parsed_path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Score
            try:
                score_data = analyze(parsed_path, pipe=pipe)
            except Exception as exc:
                logger.warning(f"{ticker} {date}: NLP failed — {exc}")
                continue

            # Save score
            score_path = SCORES_DIR / f"{ticker}_{date}_score.json"
            score_path.write_text(
                json.dumps(score_data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            rows.append(_row_from_score(score_data))

    return rows


def _row_from_score(s: dict) -> dict:
    return {
        "ticker":       s.get("ticker", "?"),
        "date":         s.get("date", "?"),
        "sections":     "Y" if s.get("sections_found") else "N",
        "source":       s.get("source", "edgar"),
        "hedge_pct":    round(s.get("hedging", {}).get("hedging_density", 0.0), 2),
        "neg_sent":     round(s.get("sentiment", {}).get("overall_negative_ratio", 0.0), 3),
        "bwd_ratio":    round(s.get("vocab", {}).get("backward_ratio", 0.0), 3),
        "traj":         round(s.get("sentiment", {}).get("sentiment_trajectory", 0.0), 3),
        "ew_score":     round(s.get("EW_Risk_Score", 0.0), 1),
        "risk_class":   s.get("risk_class", "?"),
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No rows to display.")
        return

    # Sort by EW score descending
    rows = sorted(rows, key=lambda r: r["ew_score"], reverse=True)

    hdr = (
        f"{'Ticker':<6} {'Date':<12} {'Q&A':>3} {'Source':<12} "
        f"{'Hedge%':>7} {'NegSent':>8} {'BwdRatio':>9} "
        f"{'Traj':>6} {'EWScore':>8} {'Risk':<12}"
    )
    sep = "─" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in rows:
        flag = " ◄ HIGH" if r["ew_score"] >= 65 else (" · MED" if r["ew_score"] >= 35 else "")
        print(
            f"{r['ticker']:<6} {r['date']:<12} {r['sections']:>3} {r['source']:<12} "
            f"{r['hedge_pct']:>7.2f} {r['neg_sent']:>8.3f} {r['bwd_ratio']:>9.3f} "
            f"{r['traj']:>6.3f} {r['ew_score']:>8.1f} {r['risk_class']:<12}{flag}"
        )
    print(sep)
    high = sum(1 for r in rows if r["ew_score"] >= 65)
    med  = sum(1 for r in rows if 35 <= r["ew_score"] < 65)
    low  = sum(1 for r in rows if r["ew_score"] < 35)
    print(f"\n  Total: {len(rows)} filings  |  HIGH: {high}  MED: {med}  LOW: {low}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", nargs="+", default=VOLATILE_TICKERS,
        help="Tickers to fetch (default: all 10 volatile)",
    )
    parser.add_argument("--n", type=int, default=4, help="Transcripts per ticker")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    logger.info("Loading FinBERT pipeline...")
    pipe = load_pipeline()

    logger.info(f"Processing {len(tickers)} tickers, up to {args.n} transcripts each")
    rows = run(tickers=tickers, n=args.n, pipe=pipe)

    print_table(rows)
