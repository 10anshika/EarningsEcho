"""
vocab_scorer.py
Forward- vs backward-looking vocabulary signal for earnings text.

A high backward_ratio (> 0.65) suggests management is dwelling on past
performance rather than projecting forward — a potential deflection signal.

Public API
----------
score_vocab(text: str) -> dict
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

FORWARD_LOOKING: list[str] = [
    # Action verbs — plans / projections
    "will",
    "plan",
    "plans",
    "planning",
    "expect",
    "expects",
    "expecting",
    "target",
    "targets",
    "targeting",
    "forecast",
    "forecasts",
    "forecasting",
    "project",
    "projects",
    "projecting",
    "anticipate",
    "anticipates",
    "anticipating",
    "guide",
    "guides",
    "guidance",
    # Confidence / ambition words
    "confident",
    "confidence",
    "optimistic",
    "optimism",
    "committed",
    "commit",
    "intend",
    "intends",
    "intention",
    "aim",
    "aims",
    "aspire",
    "aspires",
    "goal",
    "goals",
    "objective",
    "objectives",
    "outlook",
    "opportunity",
    "opportunities",
    "pipeline",
    "backlog",
    "momentum",
    "accelerate",
    "accelerating",
    "invest",
    "investing",
    "investment",
    "grow",
    "growing",
    "growth",
    "expand",
    "expanding",
    "expansion",
    "scale",
    "scaling",
]

BACKWARD_LOOKING: list[str] = [
    # Past tense verbs
    "was",
    "were",
    "had",
    "did",
    "achieved",
    "delivered",
    "completed",
    "reported",
    "posted",
    "recorded",
    "generated",
    "produced",
    "returned",
    "gained",
    "grew",         # past tense of grow — ambiguous but usually backward
    "declined",
    "fell",
    "dropped",
    "decreased",
    "increased",    # past tense context
    "improved",
    "reduced",
    "exceeded",
    "missed",
    "came in",
    "ended",
    "closed",
    "finished",
    "launched",
    "announced",
    "signed",
    "entered",
    "acquired",
    "divested",
    "returned",
    "paid",
    "repurchased",
    "recognized",
    "established",
    "reflected",
    "resulted",
    "contributed",
    "benefited",
    "impacted",
    "affected",
]

# Pre-compile: match whole words only
_FWD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FORWARD_LOOKING) + r")\b",
    re.IGNORECASE,
)
_BWD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BACKWARD_LOOKING) + r")\b",
    re.IGNORECASE,
)

# Threshold: backward_ratio > this is flagged as deflection
DEFLECTION_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_vocab(text: str) -> dict:
    """
    Compute forward/backward-looking vocabulary balance.

    Parameters
    ----------
    text : str — earnings narrative text

    Returns
    -------
    dict with keys:
        forward_count     — matched forward-looking words
        backward_count    — matched backward-looking words
        total_signal      — forward_count + backward_count
        forward_ratio     — forward_count / total_signal  (0 if total == 0)
        backward_ratio    — backward_count / total_signal (0 if total == 0)
        deflection_flag   — True if backward_ratio > DEFLECTION_THRESHOLD
        top_forward       — list of (word, count), top 10 forward hits
        top_backward      — list of (word, count), top 10 backward hits
    """
    if not text or not text.strip():
        return _empty_result()

    fwd_matches = _FWD_PATTERN.findall(text)
    bwd_matches = _BWD_PATTERN.findall(text)

    fwd_count = len(fwd_matches)
    bwd_count = len(bwd_matches)
    total = fwd_count + bwd_count

    if total == 0:
        return _empty_result()

    forward_ratio = fwd_count / total
    backward_ratio = bwd_count / total

    # Frequency tables
    fwd_freq: dict[str, int] = {}
    for w in fwd_matches:
        fwd_freq[w.lower()] = fwd_freq.get(w.lower(), 0) + 1
    bwd_freq: dict[str, int] = {}
    for w in bwd_matches:
        bwd_freq[w.lower()] = bwd_freq.get(w.lower(), 0) + 1

    top_forward = sorted(fwd_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    top_backward = sorted(bwd_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "forward_count": fwd_count,
        "backward_count": bwd_count,
        "total_signal": total,
        "forward_ratio": round(forward_ratio, 4),
        "backward_ratio": round(backward_ratio, 4),
        "deflection_flag": backward_ratio > DEFLECTION_THRESHOLD,
        "top_forward": top_forward,
        "top_backward": top_backward,
    }


def _empty_result() -> dict:
    """Return a zero-valued vocab result dict for empty or missing text."""
    return {
        "forward_count": 0,
        "backward_count": 0,
        "total_signal": 0,
        "forward_ratio": 0.0,
        "backward_ratio": 0.0,
        "deflection_flag": False,
        "top_forward": [],
        "top_backward": [],
    }
