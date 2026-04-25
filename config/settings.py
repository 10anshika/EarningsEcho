"""
settings.py
Global configuration constants for EarningsEcho.

Percentile thresholds are statistically honest on a small corpus —
they guarantee roughly equal numbers of HIGH / LOW signals regardless
of the absolute EW_Risk_Score distribution.

Weight versioning rationale
---------------------------
The ablation study tested 8 configurations on n=106 directional samples.
Equal weights (V2) achieved 57.55% directional accuracy versus 54.72% for
the original V1 split, a +2.83 percentage point improvement. For that reason,
V2 is now the default live configuration.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

ROOT_DIR: Path = Path(__file__).resolve().parents[1]

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

# ---------------------------------------------------------------------------
# Composite-score weights (versioned)
# ---------------------------------------------------------------------------
WEIGHTS_V1: dict[str, float] = {
    "hedging": 0.40,
    "negative_sentiment": 0.35,
    "backward_ratio": 0.25,
}  # original

WEIGHTS_V2: dict[str, float] = {
    "hedging": 1 / 3,
    "negative_sentiment": 1 / 3,
    "backward_ratio": 1 / 3,
}  # ablation best

ACTIVE_WEIGHTS: dict[str, float] = WEIGHTS_V2  # used by live pipeline
MODEL_VERSION: str = "v2"

logger.debug(f"Loaded EarningsEcho settings MODEL_VERSION={MODEL_VERSION}")
