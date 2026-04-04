"""
hedging_detector.py
Detect hedging language in earnings text across four linguistic categories.

Public API
----------
score_hedging(text: str) -> dict
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Hedging vocabulary — 4 categories, 50+ phrases total
# ---------------------------------------------------------------------------

# Category 1: Epistemic hedges — speaker's degree of certainty/belief
EPISTEMIC_HEDGES: list[str] = [
    "we believe",
    "we think",
    "we feel",
    "it appears",
    "it seems",
    "it would appear",
    "in our view",
    "in our opinion",
    "to our knowledge",
    "as far as we know",
    "we understand",
    "we consider",
    "we expect",          # intentionally in both — forward-looking AND epistemic
    "we anticipate",
    "it is our view",
    "management believes",
    "we are of the opinion",
]

# Category 2: Approximators — imprecision of quantity / degree
APPROXIMATORS: list[str] = [
    "approximately",
    "roughly",
    "around",
    "about",
    "in the range of",
    "in the vicinity of",
    "on the order of",
    "somewhere around",
    "more or less",
    "give or take",
    "up to",
    "as much as",
    "at least",
    "broadly",
    "generally",
    "largely",
    "in general",
]

# Category 3: Shields — deflect personal commitment / responsibility
SHIELDS: list[str] = [
    "subject to",
    "contingent on",
    "depending on",
    "conditional upon",
    "may",
    "might",
    "could",
    "would",
    "should",
    "difficult to predict",
    "hard to predict",
    "inherent uncertainty",
    "uncertain",
    "unpredictable",
    "difficult to estimate",
    "no assurance",
    "no guarantee",
    "cannot assure",
    "cannot guarantee",
    "we cannot predict",
    "beyond our control",
    "outside our control",
]

# Category 4: Plausibility shields — possibility without commitment
PLAUSIBILITY_SHIELDS: list[str] = [
    "possibly",
    "perhaps",
    "conceivably",
    "potentially",
    "there is a possibility",
    "there is a chance",
    "it is possible",
    "it may be",
    "it might be",
    "it could be",
    "under certain circumstances",
    "in some cases",
    "in certain situations",
]

# Compile all phrases into one lookup structure
_ALL_PHRASES: dict[str, str] = {}  # phrase -> category
for _phrase in EPISTEMIC_HEDGES:
    _ALL_PHRASES[_phrase] = "epistemic"
for _phrase in APPROXIMATORS:
    _ALL_PHRASES[_phrase] = "approximator"
for _phrase in SHIELDS:
    _ALL_PHRASES[_phrase] = "shield"
for _phrase in PLAUSIBILITY_SHIELDS:
    _ALL_PHRASES[_phrase] = "plausibility"

# Pre-compile a single regex that matches any phrase (word-boundary aware)
# Sort by length descending so longer phrases match preferentially
_SORTED_PHRASES = sorted(_ALL_PHRASES.keys(), key=len, reverse=True)
_HEDGE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _SORTED_PHRASES) + r")\b",
    re.IGNORECASE,
)

# Threshold above which we classify the document as "High Uncertainty"
HIGH_UNCERTAINTY_THRESHOLD = 1.5  # hedging_density per 100 words (recalibrated for full call transcripts; press-release era was 6.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_hedging(text: str) -> dict:
    """
    Detect hedging phrases and compute hedging density.

    Parameters
    ----------
    text : str — the text to analyse (typically opening_remarks or full doc)

    Returns
    -------
    dict with keys:
        hedge_count           — total matched hedge phrases
        word_count            — total word count in text
        hedging_density       — (hedge_count / word_count) * 100
        epistemic_count       — matches in epistemic hedge category
        approximator_count    — matches in approximator category
        shield_count          — matches in shield category
        plausibility_count    — matches in plausibility shield category
        uncertainty_flag      — True if hedging_density > HIGH_UNCERTAINTY_THRESHOLD
        top_phrases           — list of (phrase, count) sorted by frequency, top 10
    """
    if not text or not text.strip():
        return _empty_result()

    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count == 0:
        return _empty_result()

    # Find all hedge matches
    matches = _HEDGE_PATTERN.findall(text.lower())

    category_counts: dict[str, int] = {
        "epistemic": 0,
        "approximator": 0,
        "shield": 0,
        "plausibility": 0,
    }
    phrase_freq: dict[str, int] = {}

    for match in matches:
        cat = _ALL_PHRASES.get(match.lower(), "shield")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        phrase_freq[match.lower()] = phrase_freq.get(match.lower(), 0) + 1

    hedge_count = len(matches)
    hedging_density = (hedge_count / word_count) * 100

    top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "hedge_count": hedge_count,
        "word_count": word_count,
        "hedging_density": round(hedging_density, 4),
        "epistemic_count": category_counts["epistemic"],
        "approximator_count": category_counts["approximator"],
        "shield_count": category_counts["shield"],
        "plausibility_count": category_counts["plausibility"],
        "uncertainty_flag": hedging_density > HIGH_UNCERTAINTY_THRESHOLD,
        "top_phrases": top_phrases,
    }


def _empty_result() -> dict:
    """Return a zero-valued hedging result dict for empty or missing text."""
    return {
        "hedge_count": 0,
        "word_count": 0,
        "hedging_density": 0.0,
        "epistemic_count": 0,
        "approximator_count": 0,
        "shield_count": 0,
        "plausibility_count": 0,
        "uncertainty_flag": False,
        "top_phrases": [],
    }
