"""Candidate market schema.

A :class:`CandidateMarket` is a *proposed* prediction market that we want to
evaluate before (hypothetically) creating it. Nothing here touches a real
platform -- it is purely a description used by the validator, economics model
and simulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MarketStatus(str, Enum):
    """Lifecycle status of a candidate market (within this lab)."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    SIMULATED = "simulated"
    ARCHIVED = "archived"


def _parse_time(value: Any) -> datetime | None:
    """Best-effort parse of an ISO-8601 timestamp into an aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        # Allow a trailing "Z" (UTC) which fromisoformat historically rejected.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class CandidateMarket:
    """A proposed prediction market.

    Fee fields are expressed as fractions (e.g. ``0.01`` == 1%).
    Monetary fields are abstract "units" (USD-equivalent) -- the lab never
    moves real money.
    """

    id: str
    question: str
    description: str = ""
    category: str = "other"
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])
    close_time: datetime | None = None
    event_time: datetime | None = None
    resolution_source: str = ""
    platform: str = "polkamarkets"
    creator_fee: float = 0.01
    lp_fee: float = 0.02
    initial_liquidity: float = 0.0
    expected_volume: float = 0.0
    tags: list[str] = field(default_factory=list)
    status: MarketStatus = MarketStatus.PROPOSED

    def __post_init__(self) -> None:
        # Normalise times that may arrive as strings (e.g. from JSON).
        self.close_time = _parse_time(self.close_time)
        self.event_time = _parse_time(self.event_time)
        if not isinstance(self.status, MarketStatus):
            self.status = MarketStatus(str(self.status))
        # Defensive copies / coercions.
        self.outcomes = [str(o) for o in self.outcomes]
        self.tags = [str(t) for t in self.tags]
        self.creator_fee = float(self.creator_fee)
        self.lp_fee = float(self.lp_fee)
        self.initial_liquidity = float(self.initial_liquidity)
        self.expected_volume = float(self.expected_volume)

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def num_outcomes(self) -> int:
        return len(self.outcomes)

    @property
    def horizon_days(self) -> float | None:
        """Days from now (UTC) until the event resolves, if known."""
        if self.event_time is None:
            return None
        now = datetime.now(timezone.utc)
        return (self.event_time - now).total_seconds() / 86400.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateMarket":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["close_time"] = self.close_time.isoformat() if self.close_time else None
        d["event_time"] = self.event_time.isoformat() if self.event_time else None
        return d
