"""Market-selection scorer.

Answers "should I create this market?" from the LP/creator's point of view,
using the project's central finding: the market is efficient, so profit comes
from **fees beating adverse selection**, not from prediction.

The scorer estimates, per candidate market:
  * the adverse-selection cost it will face (grounded in real data, by category)
  * whether current fees clear that cost (with margin)
  * the fees + liquidity to recommend
  * a verdict: create / risky / avoid (/ blocked)

It is transparent about which inputs are data-measured vs heuristic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.models.candidate_market import CandidateMarket
from src.models.platform_profile import get_platform_profile
from src.validation.validator import validate_market


# --- Data-measured: adverse-selection magnitude by category -----------------
# avg |settle - early price| measured on REAL-MONEY markets (Polymarket, n=2,500
# with price history). Higher = the price moves more after you provide
# liquidity = more toxic flow. These supersede the earlier combined (play+real)
# numbers: a pressure test showed play-money and real-money disagree sharply,
# and a creator/LP operates in the real-money regime. Most categories now have
# large samples and tight confidence intervals (see CATEGORY_CONFIDENCE).
# Notable real-money correction: economics is fairly *safe* (0.175), not the
# worst as the combined data wrongly suggested.
CATEGORY_ADVERSE_MAGNITUDE: dict[str, float] = {
    "sports": 0.090,       # safest; tight CI [0.076, 0.104], n=504
    "economics": 0.175,    # n=98 (medium)
    "politics": 0.177,     # n=621
    "crypto": 0.254,       # n=359
    "tech": 0.254,         # n=81 (medium)
    "geopolitics": 0.269,  # n=304
    "other": 0.303,        # riskiest bucket; n=525
    # science: no real-money data -> falls back to DEFAULT.
}
DEFAULT_ADVERSE_MAGNITUDE = 0.211  # real-money overall mean

# Confidence per category, from real-money sample size + bootstrap CI width.
CATEGORY_CONFIDENCE: dict[str, str] = {
    "sports": "high",        # n=504, tight CI
    "politics": "high",      # n=621
    "crypto": "high",        # n=359
    "geopolitics": "high",   # n=304
    "other": "high",         # n=525
    "economics": "medium",   # n=98
    "tech": "medium",        # n=81
    "science": "low",        # no real-money data
}

# --- Heuristic: how sharp-dominated each category's flow tends to be ---------
# (Not data-measured -- a prior. Crypto/econ draw pros; sports draws fans.)
CATEGORY_SHARP_BASE: dict[str, float] = {
    "entertainment": 0.15,
    "sports": 0.20,
    "politics": 0.25,
    "other": 0.25,
    "geopolitics": 0.30,
    "science": 0.30,
    "tech": 0.35,
    "crypto": 0.40,
    "economics": 0.40,
}
DEFAULT_SHARP_BASE = 0.30

# Economic constants.
_CAPTURE = 0.5          # informed traders capture ~half the move (slippage/timing)
_OPS_RATE = 0.002       # ~0.2% of volume in gas/ops
_TARGET_MARGIN = 0.015  # net margin per $ volume we want before calling "create"
_BREAKEVEN_SHARP = 0.30  # ~30% informed flow is break-even at ~4% fees (measured)


@dataclass
class ScoreResult:
    market_id: str
    verdict: str  # "create" | "risky" | "avoid" | "blocked"
    score: int    # 0-100 attractiveness (higher = better LP market)
    estimated_sharp_fraction: float
    adverse_selection_rate: float   # cost per $1 of volume
    fee_rate: float                 # current total fee per $1 of volume
    net_margin_rate: float          # current net per $1 of volume
    recommended_creator_fee: float
    recommended_lp_fee: float
    net_margin_at_recommended: float
    recommended_max_liquidity: float
    reasons: list[str] = field(default_factory=list)
    confidence: str = "low"  # how much to trust the category's adverse-sel number

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reasons"] = list(self.reasons)
        return d


class MarketScorer:
    def __init__(self, target_margin: float = _TARGET_MARGIN) -> None:
        self.target_margin = target_margin

    # ------------------------------------------------------------------ #
    @staticmethod
    def _public_interest(market: CandidateMarket) -> float:
        """Rough 0..1 popularity proxy from expected volume (more = more casual)."""
        v = max(1.0, market.expected_volume)
        return min(1.0, math.log10(v) / 6.0)  # ~1.0 at $1M+ volume

    def _estimate(self, market: CandidateMarket) -> tuple[float, float]:
        """Return (sharp_fraction, adverse_selection_rate per $1 volume)."""
        cat = (market.category or "other").lower()
        magnitude = CATEGORY_ADVERSE_MAGNITUDE.get(cat, DEFAULT_ADVERSE_MAGNITUDE)
        sharp_base = CATEGORY_SHARP_BASE.get(cat, DEFAULT_SHARP_BASE)
        interest = self._public_interest(market)
        # More public interest -> more casual flow -> lower sharp fraction.
        sharp = max(0.05, min(0.85, sharp_base * (1.0 - 0.5 * interest)))
        adverse_rate = sharp * magnitude * _CAPTURE
        return sharp, adverse_rate

    def _recommend_fees(self, market: CandidateMarket,
                        adverse_rate: float) -> tuple[float, float, float]:
        """Pick fees that clear break-even + target margin, within platform caps."""
        platform = get_platform_profile(market.platform)
        needed_total = adverse_rate + _OPS_RATE + self.target_margin
        max_total = platform.max_creator_fee + platform.max_lp_fee
        rec_total = min(needed_total, max_total)
        # Keep a modest creator fee; put the rest on the LP fee (capped).
        creator = min(getattr(platform, "default_creator_fee", 0.01),
                      platform.max_creator_fee)
        lp = min(max(0.0, rec_total - creator), platform.max_lp_fee)
        net_at_rec = (creator + lp) - adverse_rate - _OPS_RATE
        return round(creator, 4), round(lp, 4), net_at_rec

    def _recommended_liquidity(self, market: CandidateMarket,
                               net_margin_rate: float) -> float:
        """Cap committed capital relative to the market's risk."""
        base = market.initial_liquidity or 0.0
        # Healthier margin -> comfortable committing more; thin/negative -> less.
        if net_margin_rate <= 0:
            return round(min(base, 500.0), 2)
        scale = 0.5 + 50.0 * net_margin_rate  # ~1.0x at 1% margin, ~2x at 3%
        return round(max(base, market.expected_volume * 0.05 * scale), 2)

    # ------------------------------------------------------------------ #
    def score(self, market: CandidateMarket) -> ScoreResult:
        validation = validate_market(market)
        if not validation.allowed:
            return ScoreResult(
                market_id=market.id, verdict="blocked", score=0,
                estimated_sharp_fraction=0.0, adverse_selection_rate=0.0,
                fee_rate=0.0, net_margin_rate=0.0,
                recommended_creator_fee=0.0, recommended_lp_fee=0.0,
                net_margin_at_recommended=0.0, recommended_max_liquidity=0.0,
                reasons=validation.reasons,
            )

        sharp, adverse_rate = self._estimate(market)
        fee_rate = market.creator_fee + market.lp_fee  # liquidity_share = 1
        net_rate = fee_rate - adverse_rate - _OPS_RATE

        rec_creator, rec_lp, net_at_rec = self._recommend_fees(market, adverse_rate)
        rec_liq = self._recommended_liquidity(market, max(net_rate, net_at_rec))

        reasons: list[str] = []
        cat = (market.category or "other").lower()
        reasons.append(
            f"category '{cat}': adverse-selection magnitude "
            f"{CATEGORY_ADVERSE_MAGNITUDE.get(cat, DEFAULT_ADVERSE_MAGNITUDE):.3f} "
            f"(measured), est. sharp flow {sharp:.0%}"
        )
        reasons.append(
            f"adverse-selection cost ~{adverse_rate:.2%} of volume vs fees "
            f"{fee_rate:.2%} -> net {net_rate:+.2%}"
        )

        # Verdict.
        if net_rate >= self.target_margin:
            verdict = "create"
            reasons.append("current fees comfortably beat adverse selection")
        elif net_at_rec > 0:
            verdict = "risky"
            if net_rate <= 0:
                reasons.append(
                    f"current fees don't clear it; raise to creator "
                    f"{rec_creator:.1%} + LP {rec_lp:.1%} for {net_at_rec:+.2%}"
                )
            else:
                reasons.append("thin margin; consider higher fees and many markets")
        else:
            verdict = "avoid"
            reasons.append(
                "adverse selection too high to clear even at max platform fees"
            )

        confidence = CATEGORY_CONFIDENCE.get(cat, "low")
        if confidence == "low":
            reasons.append(
                f"⚠ low confidence: thin real-money data for '{cat}' — "
                "treat its number as a rough guess"
            )
        else:
            reasons.append(
                f"{confidence}-confidence adverse-selection number for '{cat}' "
                "(measured on real-money markets)"
            )

        # 0-100 score from the best achievable net margin (4% net == 100).
        best_net = max(net_rate, net_at_rec)
        score = int(max(0, min(100, round(best_net / 0.04 * 100))))

        return ScoreResult(
            market_id=market.id, verdict=verdict, score=score,
            estimated_sharp_fraction=round(sharp, 4),
            adverse_selection_rate=round(adverse_rate, 4),
            fee_rate=round(fee_rate, 4), net_margin_rate=round(net_rate, 4),
            recommended_creator_fee=rec_creator, recommended_lp_fee=rec_lp,
            net_margin_at_recommended=round(net_at_rec, 4),
            recommended_max_liquidity=rec_liq, reasons=reasons,
            confidence=confidence,
        )
