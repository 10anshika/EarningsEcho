"""
transcript_parser.py
Normalize raw earnings call text (press releases or full transcripts)
into a structured JSON with opening_remarks / qa_section split.

Public API
----------
parse_transcript(source_path) -> dict
parse_all(directory, pattern="*.json") -> list[dict]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# EDGAR filing header boilerplate that appears at the very start of every
# exhibit: sequence number, filename, type labels, iXBRL markers, etc.
_HEADER_JUNK = re.compile(
    r"""
    ^                          # top of string
    (?:                        # one or more junk blocks
        \s*                    # leading whitespace / blank lines
        (?:
            EX-\d+\.\d+        # e.g. EX-99.1
          | \d{1,3}            # sequence number (1-3 digits)
          | [a-zA-Z0-9_\-]+    # filename without path
            \.(?:htm|txt|xml)  # file extension
          | Document           # literal word
          | iXBRL              # viewer tag
          | Exhibit\s+\d+\.\d+ # "Exhibit 99.1"
        )
        \s*\n
    )+
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Footer patterns — boilerplate that appears near the end of press releases
_FOOTER_PATTERNS: list[re.Pattern] = [
    re.compile(r"NOTE TO EDITORS:.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"©\s*\d{4}\s+(?:Apple|Microsoft|Google|Amazon).*", re.DOTALL),
    re.compile(r"\*\s*The\s+company\s+plans\s+to\s+hold.*earnings\s+call.*", re.IGNORECASE | re.DOTALL),
]

# Q&A section markers (full conference call transcripts)
_QA_MARKERS = re.compile(
    r"""
    (?:^|\n)                       # start of line
    \s*
    (?:
        QUESTION(?:\s*-\s*AND\s*-\s*|\s+AND\s+)ANSWER   # "Question-and-Answer" / "Question and Answer"
      | Q\s*&\s*A\s+SESSION                              # "Q&A Session"
      | Q\s*AND\s*A                                      # "Q and A"
      | QUESTIONS?\s+AND\s+ANSWERS?                      # "Questions and Answers"
      | QUESTIONS?\s*&\s*ANSWERS?                        # "Questions & Answers" (Motley Fool)
      | OPERATOR\s*:                                     # "Operator:"
    )
    \s*:?\s*(?:\n|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Financial statements / supplemental data — marks where narrative ends
# in a press release and boilerplate tables begin
_FINANCIAL_TABLE_MARKERS = re.compile(
    r"""
    (?:^|\n)
    \s*
    (?:
        CONDENSED\s+CONSOLIDATED          # "Condensed Consolidated Statements..."  (Apple)
      | CONSOLIDATED\s+STATEMENTS         # "Consolidated Statements..."
      | FINANCIAL\s+STATEMENTS            # "Financial Statements..."
      | INCOME\s+STATEMENTS               # "Income Statements" (Microsoft)
      | BALANCE\s+SHEETS?                 # "Balance Sheet(s)" (Microsoft)
      | STATEMENTS\s+OF\s+(?:OPERATIONS|INCOME|CASH) # generic statement headers
      | RECONCILIATION\s+OF               # "Reconciliation of Non-GAAP..."
      | \w[\w\s]*RECONCILIATION           # "Constant Currency Reconciliation" etc.
      | SUPPLEMENTAL\s+FINANCIAL          # "Supplemental Financial Information"
      | SELECTED\s+FINANCIAL              # "Selected Financial Data"
      | NOTES?\s+TO\s+(?:CONDENSED\s+)?CONSOLIDATED  # footnotes
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Page number patterns
_PAGE_NUMBERS = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)

# Runs of 3+ blank lines → collapse to 2
_EXCESS_BLANK = re.compile(r"\n{3,}")

# Noise lines: lone punctuation, single chars, bare $ signs
_NOISE_LINE = re.compile(r"^\s*[\$\xa0\u2019\u2018\u201c\u201d\u2022\u2013\u2014•·\-–—]\s*$")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Strip page numbers, noise lines, excess blank lines, multi-spaces."""
    text = _PAGE_NUMBERS.sub("", text)
    lines = [l for l in text.split("\n") if not _NOISE_LINE.match(l)]
    text = "\n".join(lines)
    text = _EXCESS_BLANK.sub("\n\n", text)
    lines = [re.sub(r"[ \t]{2,}", " ", l) for l in text.split("\n")]
    return "\n".join(lines).strip()


def _strip_header(text: str) -> str:
    """Remove EDGAR filing boilerplate from the very top of the text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HEADER_JUNK.sub("", text, count=1)
    text = re.sub(r"^\s*Exhibit\s+\d+\.\d+\s*\n", "", text)
    return text


def _strip_footer(text: str) -> str:
    """Remove contact/copyright/editorial boilerplate from the end of a section."""
    for pat in _FOOTER_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[: m.start()]
    return text.strip()


def _split_sections(text: str) -> tuple[str, str, bool]:
    """
    Split text into (opening_remarks, qa_section, sections_found).

    Called with header-stripped but footer-INTACT text so that financial
    table markers (which appear after the footer boilerplate in press releases)
    are still visible for the split decision.

    Strategy (in priority order):
    1. Q&A / Operator markers  → proper conference call transcript split
    2. Financial statement markers → press release narrative vs. tables
    3. 40% character fallback → last resort

    The footer is stripped from opening_remarks after the split.

    Returns
    -------
    opening_remarks : str  (footer stripped, whitespace normalised)
    qa_section      : str  (whitespace normalised; empty if not found)
    sections_found  : bool (True only for a Q&A split)
    """
    # --- Priority 1: explicit Q&A / Operator marker ---
    m = _QA_MARKERS.search(text)
    if m:
        split_pos = m.start()
        opening = _normalize_whitespace(_strip_footer(text[:split_pos]))
        qa = _normalize_whitespace(text[split_pos:])
        if opening and qa:
            logger.debug(f"Q&A marker found at char {split_pos}")
            return opening, qa, True

    # --- Priority 2: financial statement tables marker ---
    m = _FINANCIAL_TABLE_MARKERS.search(text)
    if m:
        split_pos = m.start()
        opening_raw = text[:split_pos]
        # Strip press-release footer boilerplate that lives between narrative and tables
        opening = _normalize_whitespace(_strip_footer(opening_raw))
        qa = _normalize_whitespace(text[split_pos:])
        if len(opening) >= 200:
            logger.debug(f"Financial table marker found at char {split_pos}")
            return opening, qa, False

    # --- Priority 3: apply footer strip, then 40% character fallback ---
    text_no_footer = _normalize_whitespace(_strip_footer(text))
    split_pos = int(len(text_no_footer) * 0.40)
    # Snap to nearest paragraph break
    newline_pos = text_no_footer.rfind("\n\n", 0, split_pos)
    if newline_pos > split_pos * 0.5:
        split_pos = newline_pos
    opening = text_no_footer[:split_pos].strip()
    qa = text_no_footer[split_pos:].strip()
    logger.debug(f"Using 40% fallback split at char {split_pos}")
    return opening, qa, False


def _word_count(text: str) -> int:
    """Count whitespace-delimited word tokens in text."""
    return len(re.findall(r"\b\w+\b", text))


def _extract_date(source_path: Path) -> str:
    """Pull YYYY-MM-DD from filename like AAPL_2025-10-30_accession.json."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", source_path.stem)
    return m.group(1) if m else "unknown"


def _extract_ticker(source_path: Path, fallback: Optional[str] = None) -> str:
    """Pull ticker from filename like AAPL_2025-10-30_accession.json."""
    m = re.match(r"([A-Z]+)_", source_path.stem)
    return m.group(1) if m else (fallback or "UNKNOWN")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_transcript(source_path: str | Path) -> dict:
    """
    Parse a single raw transcript JSON file saved by edgar_fetcher.

    Parameters
    ----------
    source_path : path to a .json file produced by edgar_fetcher.py

    Returns
    -------
    dict with keys:
        ticker, date, source_file, word_count,
        opening_remarks, qa_section, sections_found
    """
    source_path = Path(source_path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))

    ticker = raw.get("ticker") or _extract_ticker(source_path)
    date = raw.get("filed_date") or _extract_date(source_path)
    source = raw.get("source", "edgar")
    raw_text = raw.get("raw_text", "")

    if not raw_text.strip():
        raise ValueError(f"Empty raw_text in {source_path}")

    # Strip EDGAR header first; _split_sections handles footer internally
    prepped = _strip_header(raw_text)
    opening_remarks, qa_section, sections_found = _split_sections(prepped)
    total_words = _word_count(opening_remarks) + _word_count(qa_section)

    result = {
        "ticker": ticker,
        "date": date,
        "source": source,
        "source_file": source_path.name,
        "word_count": total_words,
        "opening_remarks": opening_remarks,
        "qa_section": qa_section,
        "sections_found": sections_found,
    }

    logger.info(
        f"{ticker} {date}: {total_words:,} words | "
        f"opening={_word_count(opening_remarks):,}w | "
        f"qa={_word_count(qa_section):,}w | "
        f"sections_found={sections_found}"
    )
    return result


def parse_all(
    directory: str | Path,
    pattern: str = "*.json",
    out_dir: Optional[str | Path] = None,
) -> list[dict]:
    """
    Parse all matching raw transcript JSON files in *directory*.

    Parameters
    ----------
    directory : directory containing edgar_fetcher output .json files
    pattern   : glob pattern to select files (default *.json, skips *_parsed.json)
    out_dir   : if given, save each parsed result as <TICKER>_<DATE>_parsed.json

    Returns
    -------
    list of parsed result dicts (one per file)
    """
    directory = Path(directory)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    files = sorted(
        p for p in directory.glob(pattern) if "_parsed" not in p.stem
    )

    if not files:
        logger.warning(f"No files matching '{pattern}' found in {directory}")
        return []

    for fp in files:
        try:
            parsed = parse_transcript(fp)
        except Exception as exc:
            logger.error(f"Failed to parse {fp.name}: {exc}")
            continue

        results.append(parsed)

        if out_dir is not None:
            out_name = f"{parsed['ticker']}_{parsed['date']}_parsed.json"
            out_path = out_dir / out_name
            out_path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.success(f"Saved -> {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and normalize earnings transcript JSON files"
    )
    parser.add_argument(
        "input",
        help="Path to a single .json file OR a directory of .json files",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write *_parsed.json output (default: same as input dir)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Filter directory mode to only files starting with TICKER_",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        parsed = parse_transcript(input_path)
        out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{parsed['ticker']}_{parsed['date']}_parsed.json"
        out_path = out_dir / out_name
        out_path.write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved -> {out_path}")
    else:
        pattern = f"{args.ticker}_*.json" if args.ticker else "*.json"
        out_dir = Path(args.out_dir) if args.out_dir else input_path
        results = parse_all(input_path, pattern=pattern, out_dir=out_dir)
        print(f"\nDone. {len(results)} file(s) parsed.")
