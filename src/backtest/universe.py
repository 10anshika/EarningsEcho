"""
universe.py
Load and query the ticker universe from config/universe.json.
"""
from __future__ import annotations
import json
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parents[2] / "config" / "universe.json"


def load_universe(path: str | Path = _DEFAULT_PATH) -> list[dict]:
    """Return list of {ticker, sector} dicts."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["universe"]


def ticker_sector_map(path: str | Path = _DEFAULT_PATH) -> dict[str, str]:
    """Return {ticker: sector} mapping from the universe JSON file."""
    return {e["ticker"]: e["sector"] for e in load_universe(path)}


def tickers(path: str | Path = _DEFAULT_PATH) -> list[str]:
    """Return the list of ticker symbols from the universe JSON file."""
    return [e["ticker"] for e in load_universe(path)]
