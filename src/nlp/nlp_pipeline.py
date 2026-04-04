"""
nlp_pipeline.py
Single entry point: run all NLP layers on a parsed transcript JSON.

Public API
----------
analyze(parsed_json_path, pipe=None) -> dict
analyze_all(directory, pattern="*_parsed.json", out_dir=None) -> list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.nlp.finbert_scorer import load_pipeline, score_text
from src.nlp.hedging_detector import score_hedging
from src.nlp.vocab_scorer import score_vocab
from src.nlp.composite_score import compute_composite


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(parsed_json_path: str | Path, pipe=None) -> dict:
    """
    Run the full NLP pipeline on a single parsed transcript JSON.

    Parameters
    ----------
    parsed_json_path : path to a *_parsed.json file from transcript_parser
    pipe             : optional pre-loaded FinBERT pipeline (loaded once if None)

    Returns
    -------
    Complete result dict combining all sub-module outputs plus composite score.
    """
    parsed_json_path = Path(parsed_json_path)
    parsed = json.loads(parsed_json_path.read_text(encoding="utf-8"))

    ticker = parsed.get("ticker", "UNKNOWN")
    date = parsed.get("date", "unknown")
    opening = parsed.get("opening_remarks", "")
    qa = parsed.get("qa_section", "")
    full_text = f"{opening}\n\n{qa}".strip()

    logger.info(f"Analyzing {ticker} {date} — {parsed.get('word_count', 0):,} words")

    # --- Load FinBERT once and reuse ---
    if pipe is None:
        pipe = load_pipeline()

    # --- Run the three NLP layers ---
    # Hedging: scored on opening_remarks (management's prepared words)
    hedge_result = score_hedging(opening)

    # Sentiment: scored across both sections for trajectory
    sentiment_result = score_text(opening, qa, pipe=pipe)

    # Vocab: scored on opening_remarks (forward/backward balance of narrative)
    vocab_result = score_vocab(opening)

    # --- Composite score ---
    composite = compute_composite(hedge_result, sentiment_result, vocab_result)

    result = {
        "ticker": ticker,
        "date": date,
        "source": parsed.get("source", "edgar"),
        "source_file": parsed_json_path.name,
        "word_count": parsed.get("word_count", 0),
        "sections_found": parsed.get("sections_found", False),
        # Sub-module outputs
        "hedging": hedge_result,
        "sentiment": sentiment_result,
        "vocab": vocab_result,
        # Composite
        **composite,
    }

    logger.success(
        f"{ticker} {date} → EW_Risk_Score={composite['EW_Risk_Score']} "
        f"({composite['risk_class']}) | "
        f"hedging={hedge_result['hedging_density']:.2f} | "
        f"neg_sent={sentiment_result['overall_negative_ratio']:.3f} | "
        f"bwd_ratio={vocab_result['backward_ratio']:.3f} | "
        f"traj={sentiment_result['sentiment_trajectory']:+.4f}"
    )

    return result


def analyze_all(
    directory: str | Path,
    pattern: str = "*_parsed.json",
    out_dir: Optional[str | Path] = None,
) -> list[dict]:
    """
    Run the NLP pipeline on all matching parsed transcript files.

    Parameters
    ----------
    directory : directory containing *_parsed.json files
    pattern   : glob pattern to match files
    out_dir   : if given, write <TICKER>_<DATE>_score.json for each result

    Returns
    -------
    list of result dicts, one per file
    """
    directory = Path(directory)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(directory.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' in {directory}")
        return []

    # Load FinBERT once and share across all files
    pipe = load_pipeline()

    results = []
    for fp in files:
        try:
            result = analyze(fp, pipe=pipe)
        except Exception as exc:
            logger.error(f"Failed to analyze {fp.name}: {exc}")
            continue

        results.append(result)

        if out_dir is not None:
            out_name = f"{result['ticker']}_{result['date']}_score.json"
            out_path = out_dir / out_name
            # Serialise top_phrases lists to be JSON-safe
            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info(f"Saved -> {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point + summary table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full NLP pipeline on parsed transcripts")
    parser.add_argument(
        "input",
        help="Path to a single *_parsed.json file OR directory of parsed files",
    )
    parser.add_argument(
        "--out-dir",
        default="data/scores",
        help="Directory to save score JSON files (default: data/scores)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if input_path.is_file():
        results = [analyze(input_path)]
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            r = results[0]
            out_name = f"{r['ticker']}_{r['date']}_score.json"
            out_path = out_dir / out_name
            out_path.write_text(
                json.dumps(r, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
    else:
        results = analyze_all(input_path, out_dir=out_dir)

    # --- Summary table ---
    if not results:
        print("No results.")
    else:
        col_w = [6, 12, 16, 20, 15, 15, 13]
        header = (
            f"{'Ticker':<{col_w[0]}}  "
            f"{'Date':<{col_w[1]}}  "
            f"{'Hedging%':<{col_w[2]}}  "
            f"{'Sent.Trajectory':<{col_w[3]}}  "
            f"{'Backwd.Ratio':<{col_w[4]}}  "
            f"{'EW_Risk_Score':<{col_w[5]}}  "
            f"{'Risk Class':<{col_w[6]}}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)
        for r in results:
            print(
                f"{r['ticker']:<{col_w[0]}}  "
                f"{r['date']:<{col_w[1]}}  "
                f"{r['hedging']['hedging_density']:>{col_w[2]}.4f}  "
                f"{r['sentiment']['sentiment_trajectory']:>{col_w[3]}.4f}  "
                f"{r['vocab']['backward_ratio']:>{col_w[4]}.4f}  "
                f"{r['EW_Risk_Score']:>{col_w[5]}.2f}  "
                f"{r['risk_class']:<{col_w[6]}}"
            )
        print(sep)
        print(f"  {len(results)} file(s) analyzed. Scores saved to {out_dir}/\n")
