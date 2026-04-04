"""
motleyfool_fetcher.py
Fetch full earnings call transcripts from Motley Fool.

URL pattern: https://www.fool.com/earnings/call-transcripts/{YYYY}/{MM}/{DD}/{slug}/
Slug pattern: {company-name}-{ticker-lower}-{q#}-{YYYY}-earnings-call-transcript

Motley Fool robots.txt (User-agent: *) does NOT disallow /earnings/call-transcripts/.
We respect that with a 500 ms inter-request delay.

Public API
----------
search_transcripts(ticker, n=4) -> list[dict]
    Returns structured dicts compatible with edgar_fetcher output.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://www.fool.com"
TRANSCRIPT_BASE = "/earnings/call-transcripts"

HEADERS = {
    "User-Agent": "EarningsEcho dev@earningsecho.com",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY = 0.5  # seconds between requests (robots.txt courtesy)

# ---------------------------------------------------------------------------
# Ticker → Motley Fool company-name slug map
# Covers our 10 volatile tickers plus the broader backtest universe.
# Format: ticker -> company slug (lowercase, hyphens, as used in URLs)
# ---------------------------------------------------------------------------
TICKER_SLUG_MAP: dict[str, str] = {
    # Target volatile tickers
    "INTC": "intel",
    "PFE":  "pfizer",
    "CVS":  "cvs-health",
    "TGT":  "target",
    "SBUX": "starbucks",
    "WFC":  "wells-fargo",
    "C":    "citigroup",
    "NKE":  "nike",
    "MCD":  "mcdonalds",
    "SLB":  "schlumberger",
    # Broader universe
    "AAPL": "apple",
    "MSFT": "microsoft",
    "NVDA": "nvidia",
    "META": "meta-platforms",
    "GOOGL":"alphabet",
    "CRM":  "salesforce",
    "ADBE": "adobe",
    "JPM":  "jpmorgan-chase",
    "BAC":  "bank-of-america",
    "GS":   "goldman-sachs",
    "MS":   "morgan-stanley",
    "BLK":  "blackrock",
    "AXP":  "american-express",
    "JNJ":  "johnson-and-johnson",
    "UNH":  "unitedhealth-group",
    "ABBV": "abbvie",
    "MRK":  "merck",
    "LLY":  "eli-lilly",
    "MDT":  "medtronic",
    "AMZN": "amazon",
    "TSLA": "tesla",
    "HD":   "home-depot",
    "NKE":  "nike",
    "COST": "costco-wholesale",
    "XOM":  "exxon-mobil",
    "CVX":  "chevron",
    "COP":  "conocophillips",
    "EOG":  "eog-resources",
    "PSX":  "phillips-66",
    "MPC":  "marathon-petroleum",
    "VLO":  "valero-energy",
}

# Quarter candidates per year — (fiscal quarter label, calendar month of call)
# Most companies report on a calendar-year basis
_QUARTER_MONTHS = {
    "q1": [4, 5],
    "q2": [7, 8],
    "q3": [10, 11],
    "q4": [1, 2],   # Q4 reported in following January/February
}

# Days around the filing to probe (the Fool article may be +0 to +2 days)
_DATE_OFFSETS = [0, 1, 2, -1, 3]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_url(year: int, month: int, day: int, slug: str) -> str:
    """Construct the canonical Motley Fool transcript URL for a given date and slug."""
    return (
        f"{BASE_URL}{TRANSCRIPT_BASE}"
        f"/{year}/{month:02d}/{day:02d}/{slug}/"
    )


def _build_slug(ticker: str, quarter: str, year: int) -> str:
    """Build the Motley Fool transcript slug from components."""
    company = TICKER_SLUG_MAP.get(ticker.upper(), ticker.lower())
    return f"{company}-{ticker.lower()}-{quarter}-{year}-earnings-call-transcript"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def _get(url: str) -> requests.Response:
    """GET with polite delay and exponential-backoff retry; raises on non-200."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    return resp


def _probe_url(year: int, month: int, day: int, slug: str) -> Optional[str]:
    """Try a URL and return it if it resolves (200), else None."""
    url = _build_url(year, month, day, slug)
    try:
        time.sleep(REQUEST_DELAY)
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code == 200 and "transcript-content" in r.text:
            return url
    except Exception:
        pass
    return None


def _find_url(ticker: str, quarter: str, call_year: int, approx_date: date) -> Optional[str]:
    """
    Try known date offsets around approx_date to locate the transcript URL.
    Falls back to probing the expected month range if approx_date misses.
    """
    slug = _build_slug(ticker, quarter, call_year)

    # First: probe around the known date
    for offset in _DATE_OFFSETS:
        d = approx_date + timedelta(days=offset)
        url = _probe_url(d.year, d.month, d.day, slug)
        if url:
            logger.debug(f"Found transcript at offset {offset:+d}: {url}")
            return url

    # Second: probe expected calendar months for this quarter
    months = _QUARTER_MONTHS.get(quarter, [])
    probe_year = call_year if quarter != "q4" else call_year + 1
    for month in months:
        for day in range(1, 32, 4):  # sample days 1,5,9,...29
            try:
                candidate = date(probe_year, month, day)
            except ValueError:
                continue
            url = _probe_url(candidate.year, candidate.month, candidate.day, slug)
            if url:
                logger.debug(f"Found transcript via month-probe: {url}")
                return url

    return None


def _extract_transcript(soup: BeautifulSoup, url: str) -> Optional[dict]:
    """
    Parse the Motley Fool transcript page into a raw_text string plus metadata.
    Returns None if the page doesn't contain a usable transcript.
    """
    container = soup.find("div", class_="article-body transcript-content")
    if not container:
        logger.debug(f"No transcript-content div found at {url}")
        return None

    # ── Extract metadata from the first few lines ──────────────────────────
    lines_all = [l.strip() for l in container.get_text(separator="\n").split("\n") if l.strip()]

    # Find ticker, date, quarter from the structured header area
    ticker_found = None
    call_date_str = None
    for i, line in enumerate(lines_all[:15]):
        # Ticker: appears as uppercase 2-5 char string alone on a line
        if re.match(r"^[A-Z]{1,5}$", line):
            ticker_found = line
        # Date: "Jan 25, 2024"
        m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", line)
        if m:
            call_date_str = m.group(1)

    # ── Strip boilerplate sections, keep only the actual transcript ─────────
    # Find "Full Conference Call Transcript" h2 if present, or "Prepared Remarks:"
    full_transcript_start = None
    for h2 in container.find_all("h2"):
        if "full conference call transcript" in h2.get_text(strip=True).lower():
            full_transcript_start = h2
            break

    if full_transcript_start:
        # Collect all siblings after this h2
        raw_parts = []
        for sib in full_transcript_start.find_next_siblings():
            raw_parts.append(sib.get_text(separator="\n"))
        raw_text = "\n".join(raw_parts)
    else:
        # Fallback: use everything after "Prepared Remarks:" text
        full_text = container.get_text(separator="\n")
        m = re.search(r"Prepared Remarks:", full_text)
        if m:
            raw_text = full_text[m.start():]
        else:
            raw_text = full_text

    # Clean up
    raw_text = re.sub(r"[ \t]{2,}", " ", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    if len(raw_text) < 200:
        return None

    return {
        "ticker_found": ticker_found,
        "call_date_str": call_date_str,
        "raw_text": raw_text,
    }


def _parse_date(date_str: Optional[str]) -> Optional[str]:
    """Convert 'Jan 25, 2024' → '2024-01-25'."""
    if not date_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(date_str.strip(), "%b %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            from datetime import datetime
            return datetime.strptime(date_str.strip(), "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Pre-built transcript catalog
# Known volatile quarters with confirmed URLs (date, quarter, fiscal year)
# These are the target transcripts for the bad-quarter backtesting goal.
# ---------------------------------------------------------------------------
TRANSCRIPT_CATALOG: dict[str, list[tuple[int, int, int, str, int]]] = {
    # (year, month, day, quarter, call_year)
    "INTC": [
        (2024, 1, 25, "q4", 2023),
        (2024, 4, 25, "q1", 2024),
        (2024, 8, 1,  "q2", 2024),
        (2023, 10, 27, "q3", 2023),
        (2025, 1, 23, "q4", 2024),
        (2023, 4, 27, "q1", 2023),
    ],
    "PFE": [
        (2024, 1, 30, "q4", 2023),
        (2024, 4, 30, "q1", 2024),
        (2024, 7, 30, "q2", 2024),
        (2024, 10, 29, "q3", 2024),
        (2023, 10, 31, "q3", 2023),
        (2023, 8, 1, "q2", 2023),
    ],
    "CVS": [
        (2024, 8, 7, "q2", 2024),
        (2024, 11, 6, "q3", 2024),
        (2024, 2, 7, "q4", 2023),
        (2025, 2, 12, "q4", 2024),
        (2024, 5, 1, "q1", 2024),
        (2023, 8, 2, "q2", 2023),
    ],
    "TGT": [
        (2023, 11, 15, "q3", 2023),
        (2024, 3, 5, "q4", 2023),
        (2024, 8, 21, "q2", 2024),
        (2024, 11, 20, "q3", 2024),
        (2023, 8, 16, "q2", 2023),
        (2024, 5, 22, "q1", 2024),
    ],
    "SBUX": [
        (2024, 10, 30, "q4", 2024),
        (2024, 7, 30, "q3", 2024),
        (2024, 4, 30, "q2", 2024),
        (2025, 1, 28, "q1", 2025),
        (2023, 11, 2, "q4", 2023),
        (2023, 8, 1, "q3", 2023),
    ],
    "WFC": [
        (2024, 1, 12, "q4", 2023),
        (2024, 4, 12, "q1", 2024),
        (2024, 7, 12, "q2", 2024),
        (2024, 10, 11, "q3", 2024),
        (2023, 7, 14, "q2", 2023),
        (2023, 10, 13, "q3", 2023),
    ],
    "C": [
        (2024, 1, 12, "q4", 2023),
        (2024, 4, 12, "q1", 2024),
        (2024, 7, 12, "q2", 2024),
        (2024, 10, 15, "q3", 2024),
        (2023, 10, 13, "q3", 2023),
        (2023, 7, 14, "q2", 2023),
    ],
    "NKE": [
        (2024, 10, 1, "q1", 2025),
        (2024, 6, 27, "q4", 2024),
        (2024, 3, 21, "q3", 2024),
        (2023, 12, 21, "q2", 2024),
        (2023, 9, 28, "q1", 2024),
        (2025, 3, 20, "q3", 2025),
    ],
    "MCD": [
        (2024, 10, 29, "q3", 2024),
        (2024, 7, 29, "q2", 2024),
        (2024, 4, 30, "q1", 2024),
        (2025, 2, 5, "q4", 2024),
        (2023, 10, 30, "q3", 2023),
        (2024, 2, 5, "q4", 2023),
    ],
    "SLB": [
        (2024, 10, 18, "q3", 2024),
        (2024, 7, 19, "q2", 2024),
        (2024, 4, 19, "q1", 2024),
        (2025, 1, 17, "q4", 2024),
        (2023, 10, 20, "q3", 2023),
        (2024, 1, 19, "q4", 2023),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_transcripts(ticker: str, n: int = 4) -> list[dict]:
    """
    Fetch up to *n* earnings call transcripts for *ticker* from Motley Fool.

    Parameters
    ----------
    ticker : str — stock ticker (e.g. 'INTC')
    n      : int — max transcripts to return

    Returns
    -------
    list[dict] with keys compatible with edgar_fetcher:
        ticker, filed_date, form_type, exhibit_url, raw_text, source
    """
    ticker = ticker.upper()
    catalog = TRANSCRIPT_CATALOG.get(ticker)
    if not catalog:
        logger.warning(f"No Motley Fool catalog entry for {ticker}; use edgar_fetcher instead")
        return []

    logger.info(f"Fetching up to {n} Motley Fool transcripts for {ticker}")
    results: list[dict] = []

    for (year, month, day, quarter, call_year) in catalog:
        if len(results) >= n:
            break

        approx = date(year, month, day)
        slug = _build_slug(ticker, quarter, call_year)
        url = _find_url(ticker, quarter, call_year, approx)

        if not url:
            logger.debug(f"{ticker} {quarter}{call_year}: transcript not found on Motley Fool")
            continue

        try:
            resp = _get(url)
        except Exception as exc:
            logger.warning(f"Could not fetch {url}: {exc}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        extracted = _extract_transcript(soup, url)
        if not extracted:
            logger.debug(f"{ticker}: could not extract transcript from {url}")
            continue

        call_date = _parse_date(extracted["call_date_str"]) or f"{year:04d}-{month:02d}-{day:02d}"

        result = {
            "ticker": ticker,
            "cik": None,
            "accession_number": slug,
            "filed_date": call_date,
            "form_type": "8-K",
            "exhibit_url": url,
            "exhibit_desc": f"Motley Fool transcript — {quarter.upper()} {call_year}",
            "raw_text": extracted["raw_text"],
            "source": "motleyfool",
        }
        results.append(result)
        logger.success(
            f"[{len(results)}/{n}] {ticker} {call_date} {quarter.upper()} {call_year} "
            f"— {len(extracted['raw_text']):,} chars"
        )

    logger.info(f"Returned {len(results)} Motley Fool transcript(s) for {ticker}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Motley Fool earnings transcripts")
    parser.add_argument("ticker", help="Ticker symbol, e.g. INTC")
    parser.add_argument("-n", type=int, default=4, help="Number of transcripts")
    parser.add_argument("--out-dir", default="data/transcripts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = search_transcripts(args.ticker, args.n)
    for t in transcripts:
        fname = (
            f"{t['ticker']}_{t['filed_date']}"
            f"_{t['accession_number'].replace('-','_')[:50]}_mf.json"
        )
        out_path = out_dir / fname
        out_path.write_text(
            json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved -> {out_path}")

    print(f"\nDone. {len(transcripts)} transcript(s) saved to {out_dir}/")
