"""Turn a TrainingExample into plain numbers a model can learn from.

This is intentionally simple and dependency-free (no torch needed) so we can
test it and so the feature set is easy to read. A richer text encoder is future
work -- for v1 we use lightweight, robust signals.

Returns a fixed-length list of floats per example, plus the label.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timedelta, timezone

from src.data.schema import TrainingExample


# --- Question-text features -------------------------------------------------
# We turn the question wording into numbers with a "hashing bag-of-words": each
# word is hashed into one of TEXT_HASH_DIM buckets and counted. No external
# model or API -- just a stable, dependency-free way to let the model learn
# from words (team names, "rate cut", "win", "before 2027", etc.).
#
# MEASURED: on the current ~334-market dataset, turning this on *hurts* (it
# overfits -- 0 dims = 78% val acc, 64 dims = 72%). The model needs far more
# data before word features pay off. So it ships OFF (0); raise it once the
# dataset is large (rough rule of thumb: hundreds of markets per text bucket).
TEXT_HASH_DIM = 0

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "will", "be", "is", "are", "to", "of", "in", "on", "by",
    "for", "and", "or", "at", "this", "that", "it", "as", "with", "from",
    "before", "after", "than", "have", "has", "do", "does", "did", "any",
}


def _stable_hash(token: str) -> int:
    """Deterministic hash (unlike Python's per-process-randomized hash())."""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if len(t) > 1 and t not in _STOPWORDS]


def text_features(text: str, dim: int = TEXT_HASH_DIM) -> list[float]:
    """Hashing bag-of-words vector for a question (length ``dim``)."""
    if dim <= 0:
        return []  # text features disabled
    vec = [0.0] * dim
    toks = _tokens(text)
    for t in toks:
        vec[_stable_hash(t) % dim] += 1.0
    total = sum(vec) or 1.0
    return [v / total for v in vec]  # normalize so length doesn't dominate


# --- Leakage guard ----------------------------------------------------------
# For a market that has already resolved, the price near the END drifts to ~0
# or ~1 because the outcome is basically known by then. Feeding that to the
# model is "peeking at the answer" (leakage). So we only use prices from the
# EARLY part of each market's life -- the first EARLY_CUTOFF of its timeline --
# and never the near-final prices. This makes the model actually forecast.
EARLY_CUTOFF = 0.5  # use only the first 50% of each market's price history


# Stable category vocabulary -> one-hot slots. "other" catches the rest.
CATEGORIES = [
    "sports", "crypto", "politics", "elections", "economics",
    "tech", "weather", "entertainment", "other",
]

FEATURE_NAMES = (
    [f"cat_{c}" for c in CATEGORIES]
    + [
        "log_liquidity",
        "log_volume",
        "horizon_days_scaled",
        "n_outcomes_scaled",
        "has_resolution_source",
        "question_len_scaled",
        "price_open_yes",        # opening price (early)
        "price_early_yes",       # price at the end of the EARLY window
        "price_early_drift",     # movement within the early window only
        "price_early_volatility",  # volatility within the early window only
    ]
    + [f"text_{i}" for i in range(TEXT_HASH_DIM)]  # hashed question words
)


def _parse(t: str | None) -> datetime | None:
    if not t:
        return None
    try:
        s = t[:-1] + "+00:00" if t.endswith("Z") else t
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _early_yes_series(ex: TrainingExample,
                      cutoff: float = EARLY_CUTOFF) -> list[float]:
    """First-outcome ('Yes') prices from the EARLY part of the market only.

    Drops the final stretch of the price history (where the price has already
    converged to the outcome), so the model can't peek at the answer. Uses
    timestamps when available; otherwise falls back to a positional split.
    """
    pts = [p for p in ex.price_series if p.prices]
    if not pts:
        return []
    times = [_parse(p.t) for p in pts]
    if len(pts) >= 2 and all(t is not None for t in times) and times[-1] > times[0]:
        span = (times[-1] - times[0]).total_seconds()
        limit = times[0] + timedelta(seconds=span * cutoff)
        early = [p.prices[0] for p, t in zip(pts, times) if t <= limit]
        return early or [pts[0].prices[0]]
    # No usable timestamps -> keep the first `cutoff` fraction by position.
    k = max(1, int(len(pts) * cutoff))
    return [p.prices[0] for p in pts[:k]]


def featurize(ex: TrainingExample) -> list[float]:
    """Build the numeric feature vector for one example."""
    cat = ex.category.lower() if ex.category else "other"
    onehot = [1.0 if cat == c else 0.0 for c in CATEGORIES]
    if sum(onehot) == 0:
        onehot[CATEGORIES.index("other")] = 1.0

    log_liq = math.log10(max(1.0, ex.liquidity))
    log_vol = math.log10(max(1.0, ex.final_volume))

    close = _parse(ex.close_time)
    event = _parse(ex.event_time)
    if close and event:
        horizon = max(0.0, (event - close).total_seconds() / 86400.0)
    else:
        horizon = 0.0
    horizon_scaled = 1.0 - math.exp(-horizon / 180.0)

    n_out = len(ex.outcomes) or 2
    n_out_scaled = min(1.0, n_out / 32.0)
    has_src = 1.0 if ex.resolution_source.strip() else 0.0
    qlen_scaled = min(1.0, len(ex.question) / 200.0)

    # EARLY prices only -- no peeking at near-final prices (see EARLY_CUTOFF).
    ys = _early_yes_series(ex)
    if ys:
        first, last = ys[0], ys[-1]
        drift = last - first
        mean = sum(ys) / len(ys)
        var = sum((y - mean) ** 2 for y in ys) / len(ys)
        vol = math.sqrt(var)
    else:
        first = last = drift = vol = 0.0

    return onehot + [
        log_liq,
        log_vol,
        horizon_scaled,
        n_out_scaled,
        has_src,
        qlen_scaled,
        first,
        last,
        drift,
        vol,
    ] + text_features(ex.question)


def label_yes(ex: TrainingExample) -> float | None:
    """Binary label: did the first outcome ('Yes') win? None if unlabeled."""
    if ex.final_outcome_index is not None:
        return 1.0 if ex.final_outcome_index == 0 else 0.0
    if ex.final_outcome and ex.outcomes:
        return 1.0 if ex.final_outcome.lower() == ex.outcomes[0].lower() else 0.0
    return None


def build_xy(examples: list[TrainingExample]) -> tuple[list[list[float]], list[float]]:
    """Featurize a labeled dataset into (X, y) for binary outcome training."""
    X, y = [], []
    for ex in examples:
        lab = label_yes(ex)
        if lab is None:
            continue
        X.append(featurize(ex))
        y.append(lab)
    return X, y
