"""Market-selection scoring.

Given the key finding that prediction markets are efficient (you can't out-
predict them), the real edge for a creator/LP is *choosing which markets to
create*: ones whose fees will outrun adverse selection. This module scores a
candidate market on exactly that.
"""

from src.scoring.market_scorer import (
    MarketScorer,
    ScoreResult,
    CATEGORY_ADVERSE_MAGNITUDE,
)

__all__ = ["MarketScorer", "ScoreResult", "CATEGORY_ADVERSE_MAGNITUDE"]
