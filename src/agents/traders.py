"""Simulated trader agents.

Each agent type captures a stylised behaviour seen in prediction markets. They
generate *simulated demand* against a market maker -- they never trade real
money.

Common attributes
-----------------
    arrival_prob       -- per-step probability the trader shows up
    bankroll           -- capital they can deploy (units)
    price_sensitivity  -- how strongly their size reacts to perceived edge
    skill              -- 0..1 forecasting skill (1 == knows true probability)
    preferred_categories -- categories they over-trade

The key behavioural difference is how each type forms its *belief* about Yes and
sizes a trade given the market price.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum


class TraderType(str, Enum):
    NOISE = "noise"
    MOMENTUM = "momentum"
    INFORMED = "informed"
    ARBITRAGE = "arbitrage"
    FAN = "fan"  # fan / narrative trader


@dataclass
class TraderAgent:
    trader_type: TraderType
    arrival_prob: float = 0.3
    bankroll: float = 200.0
    price_sensitivity: float = 1.0
    skill: float = 0.5
    preferred_categories: list[str] = field(default_factory=list)

    # Per-agent narrative lean for fan traders (+1 loves Yes, -1 loves No).
    bias: float = 0.0

    def _category_multiplier(self, category: str) -> float:
        if not self.preferred_categories:
            return 1.0
        return 1.5 if category.lower() in {
            c.lower() for c in self.preferred_categories
        } else 0.7

    def belief(
        self,
        true_p: float,
        market_p: float,
        price_history: list[float],
        rng: random.Random,
    ) -> float:
        """Return this trader's subjective P(Yes)."""
        t = self.trader_type
        if t is TraderType.INFORMED:
            # Skilled estimate: blend of truth and noise shrinking with skill.
            noise = rng.gauss(0, (1 - self.skill) * 0.15)
            return _clip(self.skill * true_p + (1 - self.skill) * market_p + noise)
        if t is TraderType.NOISE:
            # Essentially random around the current price.
            return _clip(market_p + rng.gauss(0, 0.2))
        if t is TraderType.MOMENTUM:
            # Extrapolate recent trend.
            if len(price_history) >= 2:
                trend = price_history[-1] - price_history[-min(5, len(price_history))]
            else:
                trend = 0.0
            return _clip(market_p + trend * 1.5 + rng.gauss(0, 0.05))
        if t is TraderType.ARBITRAGE:
            # Believes price should equal a slightly-better-than-market estimate
            # and only acts on clear mispricing.
            est = self.skill * true_p + (1 - self.skill) * market_p
            return _clip(est)
        if t is TraderType.FAN:
            # Narrative-driven: pulled toward their bias regardless of truth.
            pull = 0.25 * self.bias
            return _clip(market_p + pull + rng.gauss(0, 0.1))
        return market_p

    def decide(
        self,
        true_p: float,
        market_p: float,
        category: str,
        price_history: list[float],
        rng: random.Random,
    ) -> tuple[str, float] | None:
        """Decide whether to trade this step.

        Returns ``(side, notional)`` or ``None`` if the trader passes.
        """
        if rng.random() > self.arrival_prob * self._category_multiplier(category):
            return None

        belief = self.belief(true_p, market_p, price_history, rng)
        edge = belief - market_p  # +ve => thinks Yes underpriced -> buy Yes

        # Arbitrage traders need a real edge before acting.
        threshold = 0.05 if self.trader_type is TraderType.ARBITRAGE else 0.0
        if abs(edge) <= threshold:
            return None

        # Size scales with edge, sensitivity and bankroll (logistic-capped).
        raw = self.price_sensitivity * abs(edge)
        frac = 1.0 / (1.0 + math.exp(-4 * raw)) - 0.5  # 0..0.5
        notional = max(1.0, frac * 2 * self.bankroll)
        notional = min(notional, self.bankroll)

        side = "yes" if edge > 0 else "no"
        return side, notional


def _clip(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #

_DEFAULTS: dict[TraderType, dict] = {
    TraderType.NOISE: dict(arrival_prob=0.45, bankroll=120, price_sensitivity=0.5,
                           skill=0.1),
    TraderType.MOMENTUM: dict(arrival_prob=0.30, bankroll=200, price_sensitivity=1.2,
                              skill=0.35),
    TraderType.INFORMED: dict(arrival_prob=0.12, bankroll=400, price_sensitivity=2.0,
                              skill=0.85),
    TraderType.ARBITRAGE: dict(arrival_prob=0.20, bankroll=600, price_sensitivity=2.5,
                               skill=0.7),
    TraderType.FAN: dict(arrival_prob=0.35, bankroll=90, price_sensitivity=0.8,
                         skill=0.2),
}


def make_trader(
    trader_type: TraderType,
    rng: random.Random | None = None,
    **overrides,
) -> TraderAgent:
    """Create a trader with sensible defaults for its type."""
    rng = rng or random.Random()
    params = dict(_DEFAULTS[trader_type])
    params.update(overrides)
    agent = TraderAgent(trader_type=trader_type, **params)
    if trader_type is TraderType.FAN and "bias" not in overrides:
        agent.bias = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
    return agent


def default_trader_population(
    rng: random.Random | None = None,
    scale: int = 1,
) -> list[TraderAgent]:
    """A representative mix of trader types.

    ``scale`` multiplies the count of each type.
    """
    rng = rng or random.Random()
    mix = {
        TraderType.NOISE: 5,
        TraderType.MOMENTUM: 3,
        TraderType.INFORMED: 2,
        TraderType.ARBITRAGE: 1,
        TraderType.FAN: 3,
    }
    pop: list[TraderAgent] = []
    for ttype, count in mix.items():
        for _ in range(count * scale):
            pop.append(make_trader(ttype, rng))
    return pop
