"""
shap_explainer.py
SHAP-style phrase contribution analysis for the hedging detector.

Treats the hedging detector as a bag-of-words model and computes each
phrase's contribution to the final hedge_density score:

    contribution_i = (count_i / word_count) * 100 * weight_i

where weight_i is the category weight (epistemic > approximator > shield > plausibility).

Public API
----------
explain_hedge_score(hedge_result, text) -> dict
    Returns phrase contributions + a Plotly horizontal bar chart.
"""
from __future__ import annotations

import re
from typing import Optional

import plotly.graph_objects as go

from src.nlp.hedging_detector import _ALL_PHRASES, _HEDGE_PATTERN

# ---------------------------------------------------------------------------
# Category weights — epistemic hedges are the strongest signal
# (empirically, they dominate in earnings calls that later miss estimates)
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: dict[str, float] = {
    "epistemic":     1.00,
    "shield":        0.85,
    "plausibility":  0.75,
    "approximator":  0.50,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_hedge_score(hedge_result: dict, text: str) -> dict:
    """
    Compute per-phrase contributions to the hedge_density score.

    Parameters
    ----------
    hedge_result : output of hedging_detector.score_hedging()
    text         : the text that was scored (opening_remarks recommended)

    Returns
    -------
    dict with keys:
        contributions  — list of {phrase, category, count, density, weight,
                                   contribution, pct_of_total} dicts
        total_density  — hedge_density from hedge_result
        fig            — Plotly Figure (horizontal bar chart)
    """
    word_count = hedge_result.get("word_count", 0)
    total_density = hedge_result.get("hedging_density", 0.0)

    if word_count == 0 or not text:
        return {
            "contributions": [],
            "total_density": total_density,
            "fig": _empty_fig(),
        }

    # Count each phrase's occurrences in the text
    matches = _HEDGE_PATTERN.findall(text.lower())
    phrase_counts: dict[str, int] = {}
    for m in matches:
        phrase_counts[m] = phrase_counts.get(m, 0) + 1

    # Compute contributions
    contributions = []
    for phrase, count in phrase_counts.items():
        category = _ALL_PHRASES.get(phrase, "shield")
        weight = CATEGORY_WEIGHTS.get(category, 0.75)
        density = (count / word_count) * 100
        contribution = density * weight
        contributions.append({
            "phrase":          phrase,
            "category":        category,
            "count":           count,
            "density":         round(density, 5),
            "weight":          weight,
            "contribution":    round(contribution, 5),
        })

    # Sort by contribution descending
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    # Add pct_of_total (relative to weighted sum, not raw density)
    total_contribution = sum(c["contribution"] for c in contributions) or 1.0
    for c in contributions:
        c["pct_of_total"] = round(c["contribution"] / total_contribution * 100, 1)

    top10 = contributions[:10]
    fig = _bar_chart(top10, total_density)

    return {
        "contributions": contributions,
        "total_density": total_density,
        "fig": fig,
    }


# ---------------------------------------------------------------------------
# Internal — Plotly chart
# ---------------------------------------------------------------------------

_CATEGORY_COLORS: dict[str, str] = {
    "epistemic":    "#d62728",   # red — strongest signal
    "shield":       "#ff7f0e",   # orange
    "plausibility": "#9467bd",   # purple
    "approximator": "#1f77b4",   # blue — weakest
}


def _bar_chart(top10: list[dict], total_density: float) -> go.Figure:
    """Build a horizontal Plotly bar chart of top-10 hedge phrase contributions."""
    if not top10:
        return _empty_fig()

    phrases      = [c["phrase"] for c in top10]
    contributions = [c["contribution"] for c in top10]
    counts        = [c["count"] for c in top10]
    categories    = [c["category"] for c in top10]
    pcts          = [c["pct_of_total"] for c in top10]
    colors        = [_CATEGORY_COLORS.get(cat, "#aec7e8") for cat in categories]

    hover = [
        f"<b>{p}</b><br>"
        f"Category: {cat}<br>"
        f"Occurrences: {cnt}<br>"
        f"Contribution: {con:.5f}<br>"
        f"Share of hedge score: {pct:.1f}%"
        for p, cat, cnt, con, pct in zip(phrases, categories, counts, contributions, pcts)
    ]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=phrases,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        text=[f"{pct:.0f}%" for pct in pcts],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(
            text=f"What drove the hedge score  (total density: {total_density:.3f})",
            font=dict(size=15),
        ),
        xaxis_title="Weighted contribution to hedge density",
        yaxis=dict(autorange="reversed"),
        height=max(280, len(top10) * 34 + 80),
        margin=dict(t=50, b=40, l=180, r=60),
        template="plotly_white",
        showlegend=False,
    )

    # Colour legend via annotations
    for cat, color in _CATEGORY_COLORS.items():
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.0, y=-0.12 - list(_CATEGORY_COLORS.keys()).index(cat) * 0.06,
            text=f"<span style='color:{color}'>■</span> {cat}",
            showarrow=False, font=dict(size=11), xanchor="right",
        )

    return fig


def _empty_fig() -> go.Figure:
    """Return a placeholder Plotly figure when no hedging phrases were detected."""
    fig = go.Figure()
    fig.update_layout(
        title="What drove the hedge score",
        annotations=[dict(
            text="No hedging phrases detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )],
        height=200,
        template="plotly_white",
    )
    return fig
