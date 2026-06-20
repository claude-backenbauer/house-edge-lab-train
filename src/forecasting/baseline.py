"""Conservative baseline forecaster.

A deliberately simple, transparent forecaster used as a floor/sanity baseline
before any learned model exists. It does NOT call any external LLM or API
(that's explicitly out of scope for v1).

Philosophy: when uncertain, stay near the uninformative prior (uniform over
outcomes) and report *wide* uncertainty. The forecaster nudges away from the
prior only when features justify it, and it widens uncertainty for long
horizons, poor resolution quality and thin liquidity.

Inputs (:class:`ForecastFeatures`):
    category
    event_horizon_days
    resolution_quality      0..1  (1 == crisp, objective source)
    public_interest         0..1  (higher => more efficient price discovery)
    historical_volatility   0..1  placeholder
    liquidity               units placeholder
    model_uncertainty       0..1  caller-supplied extra uncertainty

Outputs (:class:`ForecastResult`):
    probabilities   list[float] (one per outcome, sums to 1)
    confidence      0..1
    uncertainty     0..1
    explanation     str
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ForecastFeatures:
    category: str = "other"
    num_outcomes: int = 2
    event_horizon_days: float = 30.0
    resolution_quality: float = 0.7
    public_interest: float = 0.5
    historical_volatility: float = 0.3  # placeholder
    liquidity: float = 1000.0  # placeholder
    model_uncertainty: float = 0.3
    # Optional weak prior signal in [0,1] for binary markets (e.g. base rate).
    prior_yes: float | None = None


@dataclass
class ForecastResult:
    probabilities: list[float]
    confidence: float
    uncertainty: float
    explanation: str
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "probabilities": list(self.probabilities),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "explanation": self.explanation,
            **self.extras,
        }


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class BaselineForecaster:
    """Conservative, explainable baseline. No external model calls."""

    # Categories where base rates are informative enough to nudge the prior.
    _CATEGORY_PRIORS = {
        "sports": 0.5,
        "elections": 0.5,
        "politics": 0.5,
        "crypto": 0.5,
        "economics": 0.5,
        "weather": 0.5,
        "tech": 0.4,  # "will X ship by date" skews No historically
    }

    def forecast(self, features: ForecastFeatures) -> ForecastResult:
        n = max(2, features.num_outcomes)
        uniform = 1.0 / n

        # --- Uncertainty assembly ---------------------------------------- #
        # Longer horizon => more uncertainty (saturating).
        horizon_term = 1.0 - math.exp(-features.event_horizon_days / 180.0)
        # Poor resolution quality => more uncertainty.
        resolution_term = 1.0 - _clip(features.resolution_quality)
        # Low public interest => thinner discovery => more uncertainty.
        interest_term = 1.0 - _clip(features.public_interest)
        # Thin liquidity => more uncertainty (log-scaled placeholder).
        liq = max(1.0, features.liquidity)
        liquidity_term = _clip(1.0 - math.log10(liq) / 4.0)  # ~0 at 10k+
        vol_term = _clip(features.historical_volatility)

        uncertainty = _clip(
            0.25 * horizon_term
            + 0.25 * resolution_term
            + 0.15 * interest_term
            + 0.15 * liquidity_term
            + 0.10 * vol_term
            + 0.10 * _clip(features.model_uncertainty)
        )
        confidence = _clip(1.0 - uncertainty)

        # --- Point estimate ---------------------------------------------- #
        # Start uniform; only binary markets get a (weak) prior nudge.
        probs = [uniform] * n
        if n == 2:
            base = features.prior_yes
            if base is None:
                base = self._CATEGORY_PRIORS.get(features.category.lower(), 0.5)
            # Shrink the prior toward 0.5 in proportion to uncertainty: the less
            # we trust ourselves, the closer to uninformative we stay.
            shrink = confidence  # 0..1
            p_yes = _clip(0.5 + (base - 0.5) * shrink)
            probs = [p_yes, 1.0 - p_yes]

        # Normalise defensively.
        total = sum(probs)
        probs = [p / total for p in probs]

        explanation = self._explain(
            features, confidence, uncertainty, probs, horizon_term,
            resolution_term, liquidity_term,
        )

        return ForecastResult(
            probabilities=probs,
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            explanation=explanation,
            extras={"method": "conservative-baseline-v1"},
        )

    @staticmethod
    def _explain(
        f: ForecastFeatures,
        confidence: float,
        uncertainty: float,
        probs: list[float],
        horizon_term: float,
        resolution_term: float,
        liquidity_term: float,
    ) -> str:
        drivers = []
        if horizon_term > 0.5:
            drivers.append(f"long horizon (~{f.event_horizon_days:.0f}d) widens it")
        if resolution_term > 0.3:
            drivers.append(
                f"resolution quality {f.resolution_quality:.2f} is imperfect"
            )
        if liquidity_term > 0.3:
            drivers.append(f"liquidity {f.liquidity:.0f} is thin")
        if f.public_interest < 0.3:
            drivers.append("low public interest limits price discovery")
        driver_text = "; ".join(drivers) if drivers else "no strong widening factors"
        return (
            f"Conservative baseline for category '{f.category}': "
            f"P={[round(p, 3) for p in probs]}, confidence={confidence:.2f}, "
            f"uncertainty={uncertainty:.2f}. The estimate stays near the "
            f"uninformative prior because {driver_text}. No external model used."
        )
