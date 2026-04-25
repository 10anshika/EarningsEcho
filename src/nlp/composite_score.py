"""
composite_score.py
Combine NLP sub-signals into an EW_Risk_Score.

Default formula
---------------
EW_Risk_Score =
    (0.40 × hedging_norm)
  + (0.35 × negative_sentiment_norm)
  + (0.25 × backward_ratio_norm)
Scaled to 0–100.

An optional sentiment-trajectory input can be included for experiments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Normalisation calibration constants
# ---------------------------------------------------------------------------
_HEDGE_CAP = 3.0           # hedging_density at which score hits 1.0
_NEG_SENT_CAP = 0.20       # negative_ratio at which score hits 1.0
_BACK_RATIO_CAP = 1.0      # backward_ratio already in [0,1]
_TRAJECTORY_CAP = 0.50     # positive trajectory above this saturates to 1.0

WEIGHTS = {
    "hedging": 0.40,
    "negative_sentiment": 0.35,
    "backward_ratio": 0.25,
}

HIGH_RISK_THRESHOLD = 65.0
LOW_RISK_THRESHOLD = 35.0


def _clip01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return max(0.0, min(float(value), 1.0))


def normalize_components(
    hedging_density: float,
    negative_ratio: float,
    backward_ratio: float,
    sentiment_trajectory: float | None = None,
) -> dict[str, float | None]:
    """
    Normalize raw signal values to [0, 1].

    For trajectory, only positive deterioration contributes to risk; negative
    trajectory is mapped to 0.0.
    """
    hedging_norm = _clip01(hedging_density / _HEDGE_CAP)
    neg_sent_norm = _clip01(negative_ratio / _NEG_SENT_CAP)
    back_ratio_norm = _clip01(backward_ratio / _BACK_RATIO_CAP)
    trajectory_norm = None
    if sentiment_trajectory is not None:
        trajectory_norm = _clip01(max(float(sentiment_trajectory), 0.0) / _TRAJECTORY_CAP)

    return {
        "hedging_norm": hedging_norm,
        "negative_sentiment_norm": neg_sent_norm,
        "backward_ratio_norm": back_ratio_norm,
        "sentiment_trajectory_norm": trajectory_norm,
    }


def compute_composite(
    hedge_result: dict,
    sentiment_result: dict,
    vocab_result: dict,
    sentiment_trajectory: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Compute EW_Risk_Score from sub-module outputs.

    Parameters
    ----------
    hedge_result          : output of hedging_detector.score_hedging()
    sentiment_result      : output of finbert_scorer.score_text()
    vocab_result          : output of vocab_scorer.score_vocab()
    sentiment_trajectory  : optional raw trajectory scalar from FinBERT
    weights               : optional custom weights dict used in experiments
    """
    active_weights = weights or WEIGHTS

    hedging_density = float(hedge_result.get("hedging_density", 0.0))
    neg_ratio = float(sentiment_result.get("overall_negative_ratio", 0.0))
    backward_ratio = float(vocab_result.get("backward_ratio", 0.0))
    trajectory_raw = sentiment_trajectory
    if trajectory_raw is None:
        trajectory_raw = sentiment_result.get("sentiment_trajectory")

    norm = normalize_components(
        hedging_density=hedging_density,
        negative_ratio=neg_ratio,
        backward_ratio=backward_ratio,
        sentiment_trajectory=trajectory_raw,
    )
    hedging_norm = float(norm["hedging_norm"])
    neg_sent_norm = float(norm["negative_sentiment_norm"])
    back_ratio_norm = float(norm["backward_ratio_norm"])
    trajectory_norm = norm["sentiment_trajectory_norm"]

    raw = (
        active_weights.get("hedging", 0.0) * hedging_norm
        + active_weights.get("negative_sentiment", 0.0) * neg_sent_norm
        + active_weights.get("backward_ratio", 0.0) * back_ratio_norm
    )
    if trajectory_norm is not None:
        raw += active_weights.get("sentiment_trajectory", 0.0) * float(trajectory_norm)
    ew_risk_score = round(raw * 100, 2)

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
        "sentiment_trajectory_norm": round(float(trajectory_norm), 4) if trajectory_norm is not None else None,
        "EW_Risk_Score": ew_risk_score,
        "risk_class": risk_class,
        "component_scores": {
            "hedging_contribution": round(active_weights.get("hedging", 0.0) * hedging_norm * 100, 2),
            "sentiment_contribution": round(
                active_weights.get("negative_sentiment", 0.0) * neg_sent_norm * 100, 2
            ),
            "vocab_contribution": round(
                active_weights.get("backward_ratio", 0.0) * back_ratio_norm * 100, 2
            ),
            "trajectory_contribution": round(
                active_weights.get("sentiment_trajectory", 0.0) * float(trajectory_norm or 0.0) * 100,
                2,
            ),
        },
    }
