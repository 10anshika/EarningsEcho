"""
edgar_fetcher.py
Fetch earnings call transcript exhibits from SEC EDGAR 8-K filings.

Public API
----------
fetch_transcripts(ticker, n=5) -> list[dict]
    Each dict contains: ticker, cik, accession_number, filed_date,
    form_type, exhibit_url, raw_text
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.sec.gov"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22earnings+call%22&dateRange=custom&startdt={start}&enddt={end}&entity={ticker}&forms=8-K"

HEADERS = {
    "User-Agent": "EarningsEcho dev@earningsecho.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
DATA_HEADERS = {
    "User-Agent": "EarningsEcho dev@earningsecho.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Exhibit descriptions that typically contain earnings call transcripts
TRANSCRIPT_KEYWORDS = re.compile(
    r"(earnings.call|conference.call|transcript|prepared.remarks)",
    re.IGNORECASE,
)

# SEC rate-limit guidance: no more than 10 requests/second
REQUEST_DELAY = 0.15  # seconds between requests


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: dict, timeout: int = 30) -> requests.Response:
    """GET with a polite delay and basic error checking."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_with_retry(url: str, headers: dict) -> requests.Response:
    """Retry wrapper around _get with exponential backoff (max 3 attempts)."""
    return _get(url, headers)


def _resolve_cik(ticker: str) -> str:
    """Map a ticker symbol to a zero-padded 10-digit CIK via EDGAR company search."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=8-K"
    # Use the company tickers JSON endpoint — faster and authoritative
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    resp = _get_with_retry(tickers_url, HEADERS)
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            logger.debug(f"Resolved {ticker} -> CIK {cik}")
            return cik
    raise ValueError(f"Could not resolve ticker '{ticker}' to a CIK on SEC EDGAR")


def _get_8k_filings(cik: str, n: int) -> list[dict]:
    """Return up to *n* 8-K filing metadata dicts from the submissions API."""
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = _get_with_retry(url, DATA_HEADERS)
    submissions = resp.json()

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filed_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for form, acc, date, doc in zip(forms, accession_numbers, filed_dates, primary_docs):
        if form == "8-K":
            filings.append(
                {
                    "accession_number": acc,
                    "filed_date": date,
                    "form_type": form,
                    "primary_doc": doc,
                }
            )
        if len(filings) >= n * 3:  # fetch extra; we'll filter to n with transcripts
            break

    # Also check older filings if they exist
    if len(filings) < n:
        older_files = submissions.get("filings", {}).get("files", [])
        for older_ref in older_files:
            if len(filings) >= n * 3:
                break
            older_url = f"https://data.sec.gov/submissions/{older_ref['name']}"
            try:
                older_resp = _get_with_retry(older_url, DATA_HEADERS)
                older_data = older_resp.json()
                o_forms = older_data.get("form", [])
                o_accs = older_data.get("accessionNumber", [])
                o_dates = older_data.get("filingDate", [])
                o_docs = older_data.get("primaryDocument", [])
                for form, acc, date, doc in zip(o_forms, o_accs, o_dates, o_docs):
                    if form == "8-K":
                        filings.append(
                            {
                                "accession_number": acc,
                                "filed_date": date,
                                "form_type": form,
                                "primary_doc": doc,
                            }
                        )
            except Exception as exc:
                logger.warning(f"Could not fetch older submissions file: {exc}")

    return filings


def _build_filing_index_url(cik: str, accession_number: str) -> str:
    """Build the EDGAR filing index URL from CIK and accession number."""
    acc_nodash = accession_number.replace("-", "")
    return f"{BASE_URL}/Archives/edgar/data/{int(cik)}/{acc_nodash}/{accession_number}-index.htm"


def _find_transcript_exhibit(cik: str, accession_number: str) -> Optional[tuple[str, str]]:
    """
    Parse the filing index page and return (exhibit_url, description) for the
    exhibit most likely to contain an earnings call transcript, or None.
    """
    index_url = _build_filing_index_url(cik, accession_number)
    try:
        resp = _get_with_retry(index_url, HEADERS)
    except requests.HTTPError as exc:
        logger.debug(f"Index fetch failed for {accession_number}: {exc}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "tableFile"})
    if table is None:
        return None

    best: Optional[tuple[str, str, int]] = None  # (url, description, priority)

    for row in table.find_all("tr")[1:]:  # skip header row
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        # EDGAR table columns: Seq | Description | Document (link) | Type | Size
        description = cells[1].get_text(strip=True)
        doc_type = cells[3].get_text(strip=True)   # e.g. "EX-99.1"
        href_tag = cells[2].find("a")
        if href_tag is None:
            continue
        href = href_tag.get("href", "")
        if not href:
            continue
        # Strip iXBRL viewer wrapper: /ix?doc=/Archives/...
        if href.startswith("/ix?doc="):
            href = href[len("/ix?doc="):]
        full_url = BASE_URL + href if href.startswith("/") else href

        # Only consider HTML/text exhibits (skip graphics, xml schema, etc.)
        if not re.search(r"\.(htm|txt)$", full_url, re.IGNORECASE):
            continue

        # Score by how likely this is a transcript / earnings content
        is_transcript = bool(TRANSCRIPT_KEYWORDS.search(description + " " + doc_type))
        is_ex99 = doc_type.startswith("EX-99")

        if is_transcript:
            priority = 10
        elif doc_type == "EX-99.1":
            priority = 7   # primary earnings exhibit — always useful
        elif is_ex99:
            priority = 4
        else:
            priority = 0

        if priority > 0 and (best is None or priority > best[2]):
            best = (full_url, description, priority)

    if best is None:
        return None
    return best[0], best[1]


def _extract_text(html: str, url: str) -> str:
    """Strip HTML tags and return clean plain text. Falls back to raw for .txt."""
    if url.lower().endswith(".txt") and not html.strip().startswith("<"):
        return html  # already plain text

    soup = BeautifulSoup(html, "lxml")
    # Remove script / style noise
    for tag in soup(["script", "style", "ix:header", "ix:nonnumeric"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_transcripts(ticker: str, n: int = 5) -> list[dict]:
    """
    Fetch up to *n* earnings call transcripts for *ticker* from SEC EDGAR.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    n : int
        Maximum number of transcripts to return (default 5).

    Returns
    -------
    list[dict]
        Each element contains:
        - ticker          : str
        - cik             : str  (zero-padded 10-digit)
        - accession_number: str
        - filed_date      : str  (YYYY-MM-DD)
        - form_type       : str  (always '8-K')
        - exhibit_url     : str
        - exhibit_desc    : str
        - raw_text        : str  (cleaned plain text of the transcript)
    """
    logger.info(f"Fetching up to {n} transcripts for {ticker}")

    cik = _resolve_cik(ticker)
    filings = _get_8k_filings(cik, n)
    logger.info(f"Found {len(filings)} 8-K filings for CIK {cik}; scanning for transcripts…")

    results: list[dict] = []

    for filing in filings:
        if len(results) >= n:
            break

        acc = filing["accession_number"]
        exhibit = _find_transcript_exhibit(cik, acc)
        if exhibit is None:
            logger.debug(f"No transcript exhibit in {acc}")
            continue

        exhibit_url, exhibit_desc = exhibit
        logger.info(f"Downloading transcript: {exhibit_url} ({exhibit_desc})")

        try:
            resp = _get_with_retry(exhibit_url, HEADERS)
        except requests.HTTPError as exc:
            logger.warning(f"Could not download exhibit {exhibit_url}: {exc}")
            continue

        raw_text = _extract_text(resp.text, exhibit_url)

        # Skip near-empty exhibits (boilerplate cover pages, etc.)
        if len(raw_text) < 300:
            logger.debug(f"Exhibit too short ({len(raw_text)} chars), skipping")
            continue

        results.append(
            {
                "ticker": ticker.upper(),
                "cik": cik,
                "accession_number": acc,
                "filed_date": filing["filed_date"],
                "form_type": filing["form_type"],
                "exhibit_url": exhibit_url,
                "exhibit_desc": exhibit_desc,
                "raw_text": raw_text,
            }
        )
        logger.success(
            f"[{len(results)}/{n}] {ticker} {filing['filed_date']} — "
            f"{len(raw_text):,} chars"
        )

    logger.info(f"Returning {len(results)} transcript(s) for {ticker}")
    return results


# ---------------------------------------------------------------------------
# CLI / test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch earnings call transcripts from EDGAR")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("-n", type=int, default=5, help="Number of transcripts to fetch")
    parser.add_argument(
        "--out-dir",
        default="data/transcripts",
        help="Directory to save JSON output",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = fetch_transcripts(args.ticker, args.n)

    for t in transcripts:
        filename = f"{t['ticker']}_{t['filed_date']}_{t['accession_number'].replace('-', '')}.json"
        out_path = out_dir / filename
        # Store everything except raw_text in metadata; raw_text in separate .txt
        meta = {k: v for k, v in t.items() if k != "raw_text"}
        payload = {**meta, "raw_text": t["raw_text"]}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved -> {out_path}")

    print(f"\nDone. {len(transcripts)} transcript(s) saved to {out_dir}/")
