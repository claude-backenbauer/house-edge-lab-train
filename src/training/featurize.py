"""Turn a TrainingExample into plain numbers a model can learn from.

This is intentionally simple and dependency-free (no torch needed) so we can
test it and so the feature set is easy to read. A richer text encoder is future
work -- for v1 we use lightweight, robust signals.

Returns a fixed-length list of floats per example, plus the label.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.data.schema import TrainingExample


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
        "price_first_yes",
        "price_last_yes",
        "price_drift",
        "price_volatility",
    ]
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


def _yes_series(ex: TrainingExample) -> list[float]:
    """Extract the first outcome's price over time (proxy for 'Yes')."""
    out = []
    for p in ex.price_series:
        if p.prices:
            out.append(p.prices[0])
    return out


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

    ys = _yes_series(ex)
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
    ]


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
