"""Training-example schema.

One :class:`TrainingExample` == one *resolved* market: everything the future
GPU model needs to learn from, plus the answer (what actually happened). This
matches the feature/target list documented in ``src/training/gpu_stub.py``.

Collect many of these into a dataset (see :class:`~src.data.store.DatasetStore`)
and hand them to your buddy's training script.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PricePoint:
    """One point in a market's price history."""

    t: str  # ISO timestamp
    prices: list[float]  # price per outcome at time t (sums to ~1)
    volume: float = 0.0  # cumulative or step volume


@dataclass
class TrainingExample:
    """A single resolved market, ready for training."""

    # --- identity / text features --- #
    market_id: str
    question: str
    description: str = ""
    category: str = "other"
    platform: str = "unknown"
    resolution_source: str = ""
    outcomes: list[str] = field(default_factory=list)

    # --- timing --- #
    close_time: str | None = None
    event_time: str | None = None

    # --- market features --- #
    price_series: list[PricePoint] = field(default_factory=list)
    final_volume: float = 0.0
    liquidity: float = 0.0

    # --- optional trader-flow features (if a platform exposes them) --- #
    trader_flow: dict[str, Any] = field(default_factory=dict)

    # --- where this row came from (always recorded, always checkable) --- #
    provenance: dict[str, Any] = field(default_factory=dict)

    # --- the label / answer --- #
    final_outcome: str | None = None
    final_outcome_index: int | None = None

    def is_labeled(self) -> bool:
        return self.final_outcome_index is not None or self.final_outcome is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingExample":
        d = dict(d)
        d["price_series"] = [
            p if isinstance(p, PricePoint) else PricePoint(**p)
            for p in d.get("price_series", [])
        ]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
