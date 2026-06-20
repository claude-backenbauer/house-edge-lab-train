"""Market validator.

Screens :class:`~src.models.candidate_market.CandidateMarket` objects for
problems that would make them a bad idea to create -- whether for *economic*
reasons (fees too high, liquidity too low) or *policy* reasons (prohibited or
high-risk categories such as death, terrorism, doxxing).

The validator is deliberately conservative: when in doubt it escalates the risk
level. Output is a :class:`ValidationResult` with:

    allowed     -- bool
    risk_level  -- low / medium / high / blocked
    reasons     -- list[str] explaining the verdict
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from src.models.candidate_market import CandidateMarket
from src.models.platform_profile import PlatformProfile, get_platform_profile


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "blocked": 3}[self.value]


@dataclass
class ValidationResult:
    allowed: bool
    risk_level: RiskLevel
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.value,
            "reasons": list(self.reasons),
        }


@dataclass
class ValidationPolicy:
    """How strict the content rules are.

    ``permissive`` (the default) reflects "trade on almost anything": topics
    that are merely distasteful/sensitive (death, illegal activity, individual
    medical outcomes, war escalation, market manipulation) are *allowed but
    flagged high-risk* instead of blocked.

    A small set of categories is ALWAYS blocked regardless of mode, because a
    market on them creates a real-world incentive to harm specific people (or
    is itself abusive): assassination, terrorism, and doxxing / targeting
    private individuals. This floor stays on in both modes by design.

    ``strict`` blocks the full original list.
    """

    mode: str = "permissive"  # "permissive" | "strict"

    @property
    def strict(self) -> bool:
        return self.mode == "strict"


# --------------------------------------------------------------------------- #
# Policy data
# --------------------------------------------------------------------------- #

# --- ALWAYS blocked (the safety floor, both modes) --------------------- #
# A market on these creates a real-world incentive to harm specific people, or
# is itself abusive. This stays on regardless of policy mode.
HARD_BLOCKED_CATEGORIES: dict[str, str] = {
    "assassination": "assassination markets are always blocked",
    "terrorism": "terrorism markets are always blocked",
    "doxxing": "doxxing / private-person targeting is always blocked",
    "private_person": "targeting a private individual is always blocked",
    "doxxing / private-person events": "doxxing / private-person targeting is always blocked",
}

HARD_BLOCKED_KEYWORDS: dict[str, str] = {
    r"\bassassinat": "assassination content is always blocked",
    r"\bterror(ism|ist)\b": "terrorism content is always blocked",
    r"\bbioweapon": "WMD/terrorism content is always blocked",
    r"\bdox(x)?(ing|ed)?\b": "doxxing content is always blocked",
    r"\bhome address\b": "private-person targeting is always blocked",
}

# --- RESTRICTED (sensitive) -------------------------------------------- #
# Blocked in strict mode; allowed-but-HIGH-risk in permissive mode.
RESTRICTED_CATEGORIES: dict[str, str] = {
    "death": "individual-death market (sensitive)",
    "war_escalation": "war-escalation market (sensitive)",
    "war escalation": "war-escalation market (sensitive)",
    "market_manipulation": "market-manipulation market (sensitive)",
    "market manipulation": "market-manipulation market (sensitive)",
    "illegal": "illegal-activity market (sensitive)",
    "illegal_activity": "illegal-activity market (sensitive)",
    "illegal activity": "illegal-activity market (sensitive)",
    "medical_individual": "individual medical-outcome market (sensitive)",
    "individual medical outcomes": "individual medical-outcome market (sensitive)",
}

RESTRICTED_KEYWORDS: dict[str, str] = {
    r"\bwill .{0,40}\bdie\b": "individual-death content (sensitive)",
    r"\bdeath of\b": "individual-death content (sensitive)",
    r"\bnuclear (strike|attack|war)\b": "war-escalation content (sensitive)",
    r"\binvade|\binvasion\b": "war-escalation content (sensitive)",
    r"\bdiagnos(ed|is)\b": "individual medical-outcome content (sensitive)",
    r"\bpump and dump\b": "market-manipulation content (sensitive)",
    r"\binsider trad": "market-manipulation / illegal content (sensitive)",
}

# Words that signal vague / ambiguous resolution criteria.
AMBIGUOUS_KEYWORDS = (
    "soon",
    "many",
    "few",
    "a lot",
    "significant",
    "significantly",
    "substantial",
    "major",
    "big",
    "huge",
    "good",
    "bad",
    "better",
    "worse",
    "popular",
    "successful",
    "reasonable",
    "appropriate",
    "etc",
    "and so on",
    "probably",
    "maybe",
    "kind of",
    "sort of",
    "some point",
)

# Phrases suggesting an outcome can't be objectively verified.
UNVERIFIABLE_KEYWORDS = (
    "believe",
    "feel",
    "feels",
    "rumor",
    "rumour",
    "secretly",
    "in private",
    "behind closed doors",
    "true intentions",
    "really thinks",
    "actually wants",
    "alien",
    "afterlife",
    "god",
)

# Economic thresholds (tunable).
EXTREME_CREATOR_FEE = 0.10  # >10% creator fee is "extreme" regardless of platform
HIGH_CREATOR_FEE = 0.05  # >5% is elevated risk
LOW_LIQUIDITY_FLOOR = 100.0  # below this, market is effectively illiquid
THIN_LIQUIDITY = 500.0  # below this, liquidity is thin (medium risk)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _scan_keywords(haystack: str, keywords: tuple[str, ...]) -> list[str]:
    found = []
    for kw in keywords:
        # word-ish boundary match
        if re.search(rf"(?<![a-z]){re.escape(kw)}", haystack):
            found.append(kw)
    return found


def validate_market(
    market: CandidateMarket,
    platform: PlatformProfile | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationResult:
    """Validate a candidate market and return a :class:`ValidationResult`."""

    if platform is None:
        platform = get_platform_profile(market.platform)
    if policy is None:
        policy = ValidationPolicy()  # permissive by default

    reasons: list[str] = []
    risk = RiskLevel.LOW
    blocked = False

    def bump(level: RiskLevel) -> None:
        nonlocal risk
        if level.rank > risk.rank:
            risk = level

    category = _normalise(market.category)
    blob = _normalise(
        " ".join(
            [
                market.question,
                market.description,
                market.category,
                " ".join(market.tags),
            ]
        )
    )

    # --- Always-blocked safety floor (both modes) ------------------------- #
    if category in HARD_BLOCKED_CATEGORIES:
        reasons.append(HARD_BLOCKED_CATEGORIES[category])
        blocked = True
    for tag in market.tags:
        t = _normalise(tag)
        if t in HARD_BLOCKED_CATEGORIES:
            reasons.append(f"tag '{tag}': {HARD_BLOCKED_CATEGORIES[t]}")
            blocked = True
    for pattern, reason in HARD_BLOCKED_KEYWORDS.items():
        if re.search(pattern, blob):
            reasons.append(reason)
            blocked = True

    # --- Restricted/sensitive categories ---------------------------------- #
    # Strict mode blocks them; permissive mode allows but flags HIGH risk.
    restricted_hit = False
    if category in RESTRICTED_CATEGORIES:
        reasons.append(RESTRICTED_CATEGORIES[category])
        restricted_hit = True
    for pattern, reason in RESTRICTED_KEYWORDS.items():
        if re.search(pattern, blob):
            reasons.append(reason)
            restricted_hit = True
    if restricted_hit:
        if policy.strict:
            blocked = True
        else:
            bump(RiskLevel.HIGH)

    # --- Structural / outcome checks -------------------------------------- #
    n = market.num_outcomes
    if n < 2:
        reasons.append(f"market has {n} outcome(s); at least 2 required")
        blocked = True
    if n > 32:
        reasons.append(f"market has {n} outcomes; maximum is 32")
        blocked = True
    if n > platform.max_outcomes:
        reasons.append(
            f"{platform.name} supports at most {platform.max_outcomes} outcomes "
            f"(got {n})"
        )
        blocked = True
    if len({_normalise(o) for o in market.outcomes}) != n:
        reasons.append("duplicate outcome labels")
        bump(RiskLevel.HIGH)

    # --- Resolution source ------------------------------------------------ #
    if not _normalise(market.resolution_source):
        reasons.append("missing resolution source")
        blocked = True

    # --- Timing ----------------------------------------------------------- #
    if market.close_time and market.event_time:
        if market.close_time > market.event_time:
            reasons.append(
                "close_time is after event_time (trading would stay open past "
                "the event)"
            )
            blocked = True
    elif market.event_time is None:
        reasons.append("missing event_time")
        bump(RiskLevel.MEDIUM)

    # --- Ambiguous wording ------------------------------------------------ #
    ambiguous = _scan_keywords(blob, AMBIGUOUS_KEYWORDS)
    if ambiguous:
        reasons.append(
            "ambiguous wording (vague terms: " + ", ".join(sorted(set(ambiguous))) + ")"
        )
        bump(RiskLevel.HIGH)
    if len(_normalise(market.question)) < 15:
        reasons.append("question is too short to be unambiguous")
        bump(RiskLevel.MEDIUM)
    if "?" not in (market.question or ""):
        reasons.append("question is not phrased as a question")
        bump(RiskLevel.MEDIUM)

    # --- Unverifiable outcomes -------------------------------------------- #
    unverifiable = _scan_keywords(blob, UNVERIFIABLE_KEYWORDS)
    if unverifiable:
        reasons.append(
            "outcome may be unverifiable (terms: "
            + ", ".join(sorted(set(unverifiable)))
            + ")"
        )
        bump(RiskLevel.HIGH)

    # --- Fees ------------------------------------------------------------- #
    if market.creator_fee > EXTREME_CREATOR_FEE:
        reasons.append(
            f"extreme creator_fee {market.creator_fee:.2%} (>{EXTREME_CREATOR_FEE:.0%})"
        )
        blocked = True
    elif market.creator_fee > HIGH_CREATOR_FEE:
        reasons.append(f"elevated creator_fee {market.creator_fee:.2%}")
        bump(RiskLevel.MEDIUM)
    if market.creator_fee < 0 or market.lp_fee < 0:
        reasons.append("negative fee")
        blocked = True

    for problem in platform.fee_within_limits(market.creator_fee, market.lp_fee):
        reasons.append(problem)
        bump(RiskLevel.HIGH)

    # --- Liquidity -------------------------------------------------------- #
    if platform.requires_initial_liquidity and market.initial_liquidity <= 0:
        reasons.append(f"{platform.name} requires initial liquidity (got 0)")
        blocked = True
    if 0 < market.initial_liquidity < LOW_LIQUIDITY_FLOOR:
        reasons.append(
            f"liquidity {market.initial_liquidity:.0f} below floor "
            f"{LOW_LIQUIDITY_FLOOR:.0f} (effectively illiquid)"
        )
        bump(RiskLevel.HIGH)
    elif LOW_LIQUIDITY_FLOOR <= market.initial_liquidity < THIN_LIQUIDITY:
        reasons.append(f"thin liquidity {market.initial_liquidity:.0f}")
        bump(RiskLevel.MEDIUM)

    # --- Volume sanity ---------------------------------------------------- #
    if market.expected_volume <= 0:
        reasons.append("expected_volume is zero or unset")
        bump(RiskLevel.MEDIUM)

    # --- Final verdict ---------------------------------------------------- #
    if blocked:
        risk = RiskLevel.BLOCKED

    if not reasons:
        reasons.append("no issues detected")

    allowed = not blocked
    return ValidationResult(allowed=allowed, risk_level=risk, reasons=reasons)
