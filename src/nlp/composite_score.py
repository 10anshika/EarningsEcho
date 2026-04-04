"""
composite_score.py
Combine hedging, sentiment, and vocab signals into the EW_Risk_Score.

Formula
-------
EW_Risk_Score = (0.40 × hedging_norm) + (0.35 × negative_sentiment_norm) + (0.25 × backward_ratio_norm)
Scaled to 0–100.

Risk classes
------------
> 65  → High Risk
35–65 → Medium Risk
< 35  → Low Risk

Public API
----------
compute_composite(hedge_result, sentiment_result, vocab_result) -> dict
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Normalisation calibration constants
# Chosen to map typical earnings-call ranges to roughly 0–1.
# Hedging density: ~0 (boilerplate) to ~15+ (very hedgy);  mid ≈ 6.5
# Negative sentiment ratio: 0–1; typical range 0.05–0.40
# Backward ratio: 0–1; deflection threshold 0.65
# ---------------------------------------------------------------------------

# Clamp inputs to [0, CAP] then divide by CAP → [0, 1]
# Recalibrated from 33 real Motley Fool call transcripts:
#   hedging_density observed range: 0.43–1.30  → cap at 3.0 (2.3σ headroom)
#   neg_sentiment   observed range: 0.028–0.166 → cap at 0.20 (spreads full range)
#   backward_ratio  observed range: 0.18–0.617  → cap stays 1.0 (already 0–1)
_HEDGE_CAP = 3.0           # hedging_density at which score hits 1.0 (was 15.0)
_NEG_SENT_CAP = 0.20       # negative_ratio at which score hits 1.0 (was 0.60)
_BACK_RATIO_CAP = 1.0      # backward_ratio already in [0,1]

WEIGHTS = {
    "hedging": 0.40,
    "negative_sentiment": 0.35,
    "backward_ratio": 0.25,
}

HIGH_RISK_THRESHOLD = 65.0
LOW_RISK_THRESHOLD = 35.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_composite(
    hedge_result: dict,
    sentiment_result: dict,
    vocab_result: dict,
) -> dict:
    """
    Compute the EW_Risk_Score from the three sub-module outputs.

    Parameters
    ----------
    hedge_result      : output of hedging_detector.score_hedging()
    sentiment_result  : output of finbert_scorer.score_text()
    vocab_result      : output of vocab_scorer.score_vocab()

    Returns
    -------
    dict with keys:
        hedging_norm          — normalised hedging component [0,1]
        negative_sentiment_norm — normalised negative sentiment component [0,1]
        backward_ratio_norm   — normalised backward ratio component [0,1]
        EW_Risk_Score         — composite score 0–100
        risk_class            — 'High Risk' | 'Medium Risk' | 'Low Risk'
        component_scores      — dict of weighted component contributions
    """
    # --- Extract raw signals ---
    hedging_density = hedge_result.get("hedging_density", 0.0)
    # Use overall negative ratio (covers both opening and Q&A)
    neg_ratio = sentiment_result.get("overall_negative_ratio", 0.0)
    backward_ratio = vocab_result.get("backward_ratio", 0.0)

    # --- Normalise to [0, 1] ---
    hedging_norm = min(hedging_density / _HEDGE_CAP, 1.0)
    neg_sent_norm = min(neg_ratio / _NEG_SENT_CAP, 1.0)
    back_ratio_norm = min(backward_ratio / _BACK_RATIO_CAP, 1.0)

    # --- Weighted sum → scale to 0–100 ---
    raw = (
        WEIGHTS["hedging"] * hedging_norm
        + WEIGHTS["negative_sentiment"] * neg_sent_norm
        + WEIGHTS["backward_ratio"] * back_ratio_norm
    )
    ew_risk_score = round(raw * 100, 2)

    # --- Risk classification ---
    if ew_risk_score > HIGH_RISK_THRESHOLD:
        risk_class = "High Risk"
    elif ew_risk_score >= LOW_RISK_THRESHOLD:
        risk_class = "Medium Risk"
    else:
        risk_class = "Low Risk"

    return {
        "hedging_norm": round(hedging_norm, 4),
        "negative_sentiment_norm": round(neg_sent_norm, 4),
        "backward_ratio_norm": round(back_ratio_norm, 4),
        "EW_Risk_Score": ew_risk_score,
        "risk_class": risk_class,
        "component_scores": {
            "hedging_contribution": round(WEIGHTS["hedging"] * hedging_norm * 100, 2),
            "sentiment_contribution": round(
                WEIGHTS["negative_sentiment"] * neg_sent_norm * 100, 2
            ),
            "vocab_contribution": round(
                WEIGHTS["backward_ratio"] * back_ratio_norm * 100, 2
            ),
        },
    }
