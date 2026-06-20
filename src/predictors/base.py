"""Predictor interface and built-in predictors.

Every predictor takes a :class:`~src.models.candidate_market.CandidateMarket`
and returns a :class:`Prediction`. That's the whole contract -- simple on
purpose, so a trained model or an external agent swarm (MiroFish) can drop in
later without changing anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable

from src.forecasting.baseline import BaselineForecaster, ForecastFeatures
from src.models.candidate_market import CandidateMarket


@dataclass
class Prediction:
    """A predictor's view of a market.

    ``probabilities`` has one entry per outcome and sums to ~1.
    """

    market_id: str
    outcomes: list[str]
    probabilities: list[float]
    confidence: float = 0.5
    uncertainty: float = 0.5
    source: str = "unknown"
    explanation: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def top_outcome(self) -> tuple[str, float]:
        i = max(range(len(self.probabilities)), key=lambda j: self.probabilities[j])
        return self.outcomes[i], self.probabilities[i]

    def prob_for(self, outcome: str) -> float | None:
        for o, p in zip(self.outcomes, self.probabilities):
            if o.lower() == outcome.lower():
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "outcomes": list(self.outcomes),
            "probabilities": list(self.probabilities),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "source": self.source,
            "explanation": self.explanation,
            "created_at": self.created_at,
        }


@runtime_checkable
class Predictor(Protocol):
    """Anything that can predict a market's outcome probabilities."""

    name: str

    def predict(self, market: CandidateMarket) -> Prediction:
        ...


class BaselinePredictor:
    """Wraps the conservative baseline forecaster as a :class:`Predictor`."""

    name = "baseline"

    def __init__(self, forecaster: BaselineForecaster | None = None) -> None:
        self._f = forecaster or BaselineForecaster()

    def predict(self, market: CandidateMarket) -> Prediction:
        horizon = market.horizon_days or 30.0
        features = ForecastFeatures(
            category=market.category,
            num_outcomes=market.num_outcomes,
            event_horizon_days=max(0.0, horizon),
            liquidity=market.initial_liquidity or 1000.0,
        )
        r = self._f.forecast(features)
        return Prediction(
            market_id=market.id,
            outcomes=list(market.outcomes),
            probabilities=list(r.probabilities),
            confidence=r.confidence,
            uncertainty=r.uncertainty,
            source=self.name,
            explanation=r.explanation,
        )


class ExternalPredictor:
    """Adapter for an external prediction source (e.g. a MiroFish agent swarm).

    You supply a function that, given a market, returns a probability list (one
    per outcome). This is the integration point for MiroFish or any other
    service -- the lab never calls it directly itself; you wire it up.

    Example
    -------
        def mirofish(market):
            # call your MiroFish deployment, return [p_yes, p_no]
            ...
        predictor = ExternalPredictor("mirofish", mirofish)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[CandidateMarket], list[float]],
        confidence: float = 0.5,
    ) -> None:
        self.name = name
        self._fn = fn
        self._confidence = confidence

    def predict(self, market: CandidateMarket) -> Prediction:
        probs = list(self._fn(market))
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]
        return Prediction(
            market_id=market.id,
            outcomes=list(market.outcomes),
            probabilities=probs,
            confidence=self._confidence,
            uncertainty=round(1.0 - self._confidence, 4),
            source=self.name,
            explanation=f"External predictor '{self.name}'.",
        )
