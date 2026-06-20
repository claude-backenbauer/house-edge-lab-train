"""Predictors.

A *predictor* is anything that looks at a market and outputs probabilities for
its outcomes. They all share one shape (:class:`Predictor`), so we can swap
between them freely:

  * :class:`BaselinePredictor`  -- the simple, conservative forecaster (built in)
  * a future GPU-trained model  -- see ``src/training/``
  * a MiroFish-style agent swarm -- plugged in via :class:`ExternalPredictor`

This common shape is the seam that lets the rest of the lab (sizing bets,
telemetry, reports) stay the same no matter which brain is doing the predicting.
"""

from src.predictors.base import (
    Prediction,
    Predictor,
    BaselinePredictor,
    ExternalPredictor,
)

__all__ = [
    "Prediction",
    "Predictor",
    "BaselinePredictor",
    "ExternalPredictor",
]
