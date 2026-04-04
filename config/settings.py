"""
settings.py
Global configuration constants for EarningsEcho.

Percentile thresholds are statistically honest on a small corpus —
they guarantee roughly equal numbers of HIGH / LOW signals regardless
of the absolute EW_Risk_Score distribution.
"""

# ---------------------------------------------------------------------------
# Signal assignment — percentile-based
# ---------------------------------------------------------------------------

# Top PERCENTILE_HIGH % of EW_Risk_Score → NEGATIVE (predict stock falls)
PERCENTILE_HIGH: int = 80

# Bottom PERCENTILE_LOW % of EW_Risk_Score → POSITIVE (predict stock rises)
PERCENTILE_LOW: int = 20

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

PRIMARY_WINDOW: str = "5d"   # headline return window: '1d', '3d', or '5d'
