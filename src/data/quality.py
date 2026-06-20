"""Data-quality sweep.

The user's standing rule: **only reliable data, swept every time**. This module
decides whether a raw market is trustworthy enough to keep, and explains *why*
anything is dropped (no silent culling).

A row must clear several bars to be kept:
  * a real, substantive question (not "Hi", an emoji, or a test market);
  * genuine engagement (enough unique participants and volume);
  * a clean, unambiguous YES/NO resolution.

Returns (kept: bool, reason: str). ``reason`` is "ok" when kept, otherwise the
specific failure -- so collectors can report a drop breakdown.
"""

from __future__ import annotations

import re

# Words made of letters (so emoji / punctuation-only questions fail).
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Obvious test/junk phrasings seen on open platforms.
_JUNK_EXACT = {
    "hi", "hello", "test", "testing", "hey", "yo", "asdf", "abc", "ok",
    "okay", "ping", "hello world", "delete me", "ignore",
}


# Self-referential / meta markets ("will this market...", trader-count bets).
_META_RE = re.compile(
    r"this market|this question|unique traders|number of (unique )?traders|"
    r"will this|resolve(s)? (yes|no)\b|how many traders|n/?a by",
    re.IGNORECASE,
)
# Personal markets ("will I...", "am I...", "will my...").
_PERSONAL_RE = re.compile(r"^(will|am|do|did|should|can|is)\s+(i|my)\b", re.IGNORECASE)

# Topic keyword vocabulary -> category. First match wins.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "politics": ("election", "president", "senate", "congress", "vote", "poll",
                 "governor", "parliament", "prime minister", "referendum",
                 "trump", "biden", "primary", "impeach"),
    "economics": ("fed ", "interest rate", "inflation", "gdp", "recession",
                  "unemployment", "stock", "s&p", "nasdaq", "earnings",
                  "jobs report", "cpi", "rate cut", "rate hike", "tariff"),
    "crypto": ("bitcoin", "btc", "ethereum", "eth ", "crypto", "solana",
               "token", "coinbase", "binance", "stablecoin", "$btc", "altcoin"),
    "sports": ("world cup", "champions league", "nba", "nfl", "super bowl",
               "premier league", "olympics", "fifa", "uefa", "championship",
               "win the", "the final", "grand prix", "playoffs", "f1"),
    "tech": ("openai", "gpt", "ai model", "spacex", "tesla", "iphone",
             "chatgpt", "llm", "nvidia", "apple", "google", "release",
             "launch", "robot", "self-driving"),
    "science": ("nasa", "vaccine", "fda", "climate", "temperature", "covid",
                "disease", "approval", "study finds", "fusion"),
    "geopolitics": ("war", "ceasefire", "treaty", "sanctions", "nato",
                    "ukraine", "russia", "china", "iran", "israel", "peace",
                    "invasion", "gaza", "taiwan"),
}

_YEAR_RE = re.compile(r"\b(19|20)\d\d\b")


def infer_topic(text: str) -> str:
    """Guess a market's topic from its text (returns 'other' if unclear)."""
    t = (text or "").lower()
    for category, words in TOPIC_KEYWORDS.items():
        if any(w in t for w in words):
            return category
    return "other"


def is_forecastable_event(question: str) -> tuple[bool, str]:
    """Keep only substantive real-world events; drop meta/personal/off-topic.

    Returns (keep, reason).
    """
    q = (question or "").strip()
    if _META_RE.search(q):
        return False, "meta-market"
    if _PERSONAL_RE.search(q):
        return False, "personal-market"
    topic = infer_topic(q)
    if topic == "other" and not _YEAR_RE.search(q):
        # No recognizable topic and no dated event -> probably not useful.
        return False, "off-topic"
    return True, "ok"


def question_looks_real(question: str, min_chars: int = 20,
                        min_words: int = 4) -> bool:
    q = (question or "").strip()
    if len(q) < min_chars:
        return False
    if q.lower() in _JUNK_EXACT:
        return False
    words = _WORD_RE.findall(q)
    if len(words) < min_words:
        return False
    # Must contain at least one "question-ish" signal.
    ql = q.lower()
    if "?" not in q and not ql.startswith(("will ", "is ", "are ", "does ",
                                           "did ", "has ", "have ", "can ")):
        return False
    return True


def manifold_reliable(
    m: dict,
    min_unique_traders: int = 15,
    min_volume: float = 250.0,
    serious_only: bool = True,
) -> tuple[bool, str]:
    """Reliability gate for a raw Manifold market dict."""
    if m.get("outcomeType") != "BINARY":
        return False, "not-binary"
    if not m.get("isResolved"):
        return False, "unresolved"
    if m.get("resolution") not in ("YES", "NO"):
        return False, "ambiguous-resolution"  # MKT / CANCEL / N-A
    if not question_looks_real(m.get("question", "")):
        return False, "low-quality-question"
    unique = m.get("uniqueBettorCount")
    if unique is not None and unique < min_unique_traders:
        return False, "too-few-traders"
    if float(m.get("volume", 0.0) or 0.0) < min_volume:
        return False, "low-volume"
    if serious_only:
        ok, reason = is_forecastable_event(m.get("question", ""))
        if not ok:
            return False, reason
    return True, "ok"


def polymarket_reliable(
    outcomes: list[str],
    prices: list[float],
    volume: float,
    question: str,
    min_volume: float = 1000.0,
) -> tuple[bool, str]:
    """Reliability gate for a parsed Polymarket market."""
    if len(outcomes) != 2 or len(prices) != 2:
        return False, "not-binary"
    if not any(p >= 0.99 for p in prices):
        return False, "ambiguous-resolution"
    if not question_looks_real(question):
        return False, "low-quality-question"
    if volume < min_volume:
        return False, "low-volume"
    ok, reason = is_forecastable_event(question)
    if not ok:
        return False, reason
    return True, "ok"
