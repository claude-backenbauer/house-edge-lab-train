"""Platform profiles.

A :class:`PlatformProfile` is a *static description* of a prediction-market
platform's publicly documented mechanics. It is used to sanity-check candidate
markets (e.g. is the creator fee within the platform's allowed range?) and to
parameterise the economics model.

No profile performs any network I/O. In particular
:class:`PolkamarketsProfile` does NOT call the Polkamarkets API -- the numbers
below are taken from public documentation and are clearly marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlatformProfile:
    """Describes what a prediction-market platform allows.

    Fee fields are fractions (``0.05`` == 5%).
    """

    name: str
    allows_market_creation: bool = False
    supports_creator_fee: bool = False
    max_creator_fee: float = 0.0
    supports_lp_fee: bool = False
    max_lp_fee: float = 0.0
    requires_token_to_create: bool = False
    requires_initial_liquidity: bool = False
    supports_api: bool = False
    notes: str = ""

    # Optional structural limits, useful for validation.
    max_outcomes: int = 32
    min_outcomes: int = 2

    def fee_within_limits(self, creator_fee: float, lp_fee: float) -> list[str]:
        """Return human-readable problems with the supplied fees (empty == ok)."""
        problems: list[str] = []
        if creator_fee > 0 and not self.supports_creator_fee:
            problems.append(f"{self.name} does not support creator fees")
        elif creator_fee > self.max_creator_fee:
            problems.append(
                f"creator_fee {creator_fee:.2%} exceeds {self.name} max "
                f"{self.max_creator_fee:.2%}"
            )
        if lp_fee > 0 and not self.supports_lp_fee:
            problems.append(f"{self.name} does not support LP fees")
        elif lp_fee > self.max_lp_fee:
            problems.append(
                f"lp_fee {lp_fee:.2%} exceeds {self.name} max {self.max_lp_fee:.2%}"
            )
        return problems


@dataclass
class PolkamarketsProfile(PlatformProfile):
    """Polkamarkets profile based on public documentation.

    Documented mechanics modelled here (public docs / whitepaper):

      * Market creation is available to POLK holders.
      * Default creator fee is 1%, configurable up to 5%.
      * Market creators must seed the market with liquidity.
      * Markets are immutable once created.
      * Outcome shares are priced between 0 and 1 and must sum to 1
        (LMSR-style automated market maker).
      * Up to 32 outcomes are supported.
      * Liquidity providers can earn fees up to 5%.

    These values are baked in as defaults but remain overridable. This class
    performs NO live API calls -- live integration is intentionally deferred.
    """

    name: str = "polkamarkets"
    allows_market_creation: bool = True
    supports_creator_fee: bool = True
    max_creator_fee: float = 0.05  # up to 5%
    supports_lp_fee: bool = True
    max_lp_fee: float = 0.05  # LPs up to 5%
    requires_token_to_create: bool = True  # POLK holders
    requires_initial_liquidity: bool = True  # creators must add liquidity
    supports_api: bool = False  # live API deliberately not implemented in v1
    max_outcomes: int = 32
    min_outcomes: int = 2

    default_creator_fee: float = 0.01  # 1% default
    immutable_after_creation: bool = True
    outcomes_sum_to_one: bool = True

    notes: str = field(
        default=(
            "Public-docs profile. Creator fee default 1% (max 5%); creators "
            "must add liquidity; markets immutable after creation; outcome "
            "prices in [0,1] summing to 1; up to 32 outcomes; LP fees up to 5%. "
            "No live API in v1 -- simulation only."
        )
    )


# A small registry so callers can look profiles up by name.
_REGISTRY: dict[str, PlatformProfile] = {
    "polkamarkets": PolkamarketsProfile(),
    # A neutral generic profile for markets that don't target a real platform.
    "generic": PlatformProfile(
        name="generic",
        allows_market_creation=True,
        supports_creator_fee=True,
        max_creator_fee=0.10,
        supports_lp_fee=True,
        max_lp_fee=0.10,
        notes="Generic permissive profile for sandbox experiments.",
    ),
}


def get_platform_profile(name: str | None) -> PlatformProfile:
    """Look up a platform profile by name (case-insensitive).

    Falls back to the ``generic`` profile for unknown platforms.
    """
    key = (name or "generic").strip().lower()
    return _REGISTRY.get(key, _REGISTRY["generic"])
