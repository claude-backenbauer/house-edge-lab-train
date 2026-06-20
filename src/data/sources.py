"""Data-source registry (reference only -- no live calls).

Where we can pull data to build the training database. This module just
*describes* the options; it never connects to anything. Pick sources, get any
needed (free) API keys, and we'll wire up read-only collectors next.

Two kinds of data we need:

  1. RESOLVED PREDICTION MARKETS -- the core training corpus: a question, its
     price history, volume, and the final outcome. This is what teaches the
     model to forecast and to estimate volume / adverse selection.

  2. GROUND-TRUTH EVENT DATA -- e.g. football results and bookmaker odds, used
     both to resolve markets and as features (the odds are a strong baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataSource:
    name: str
    kind: str  # "markets" | "sports" | "odds" | "news"
    url: str
    free: bool
    needs_key: bool
    provides: list[str] = field(default_factory=list)
    notes: str = ""


DATA_SOURCES: list[DataSource] = [
    # --- Resolved prediction markets (the main training corpus) --------- #
    DataSource(
        name="Manifold Markets API",
        kind="markets",
        url="https://docs.manifold.markets/api",
        free=True,
        needs_key=False,
        provides=["question text", "price history", "volume", "resolution"],
        notes=(
            "BEST starting corpus. Thousands of already-resolved markets with "
            "full price history and outcomes, free and key-free. Play-money, but "
            "perfect labeled data to train and calibrate on."
        ),
    ),
    DataSource(
        name="Polymarket (Gamma API / subgraph)",
        kind="markets",
        url="https://docs.polymarket.com/",
        free=True,
        needs_key=False,
        provides=["question text", "price history", "volume", "resolution"],
        notes=(
            "Real-money resolved markets -> realistic volume & adverse selection. "
            "Read-only research use; we never place orders."
        ),
    ),
    DataSource(
        name="Kalshi API",
        kind="markets",
        url="https://trading-api.readme.io/",
        free=True,
        needs_key=True,
        provides=["question text", "price history", "volume", "resolution"],
        notes="Regulated US event markets. Read-only for research.",
    ),
    DataSource(
        name="Polkamarkets subgraph",
        kind="markets",
        url="https://www.polkamarkets.com/",
        free=True,
        needs_key=False,
        provides=["question text", "price history", "volume", "resolution"],
        notes="Matches our PolkamarketsProfile. Read-only.",
    ),
    # --- Sports ground truth + odds (for the World Cup & beyond) -------- #
    DataSource(
        name="football-data.org",
        kind="sports",
        url="https://www.football-data.org/",
        free=True,
        needs_key=True,
        provides=["fixtures", "results", "competitions", "teams"],
        notes=(
            "Free tier covers major competitions incl. the World Cup. Good for "
            "resolving sports markets and as schedule/result features."
        ),
    ),
    DataSource(
        name="The Odds API",
        kind="odds",
        url="https://the-odds-api.com/",
        free=True,
        needs_key=True,
        provides=["bookmaker odds", "implied probabilities"],
        notes=(
            "Bookmaker odds across many sports. Implied probabilities are a very "
            "strong baseline feature to compare our model against."
        ),
    ),
    DataSource(
        name="API-FOOTBALL (api-sports.io)",
        kind="sports",
        url="https://www.api-football.com/",
        free=True,
        needs_key=True,
        provides=["fixtures", "results", "lineups", "stats", "odds"],
        notes="Deep football data incl. World Cup; generous free tier.",
    ),
    DataSource(
        name="TheSportsDB",
        kind="sports",
        url="https://www.thesportsdb.com/api.php",
        free=True,
        needs_key=False,
        provides=["fixtures", "results", "teams"],
        notes="Community sports DB; handy free/no-key fallback.",
    ),
    # --- News / context (for agent-style predictors like MiroFish) ------ #
    DataSource(
        name="GDELT",
        kind="news",
        url="https://www.gdeltproject.org/",
        free=True,
        needs_key=False,
        provides=["global news events", "tone", "entities"],
        notes="Bulk world-news signal; useful context for agent swarms.",
    ),
]


def recommended_sources() -> dict[str, list[DataSource]]:
    """Curated picks: what to start with for the model and for the World Cup."""
    by_name = {s.name: s for s in DATA_SOURCES}
    return {
        "train_the_model_first": [
            by_name["Manifold Markets API"],
            by_name["Polymarket (Gamma API / subgraph)"],
        ],
        "world_cup_2026": [
            by_name["football-data.org"],
            by_name["The Odds API"],
        ],
    }


def print_sources() -> None:
    """Print the registry and recommendations in plain language."""
    print("Data sources (reference only -- nothing is contacted):\n")
    for s in DATA_SOURCES:
        key = "needs free key" if s.needs_key else "no key needed"
        print(f"- {s.name}  [{s.kind}, {key}]")
        print(f"    {s.url}")
        print(f"    provides: {', '.join(s.provides)}")
        if s.notes:
            print(f"    note: {s.notes}")
        print()
    print("Recommended starting points:")
    for bucket, items in recommended_sources().items():
        print(f"  {bucket}:")
        for s in items:
            print(f"    - {s.name} ({s.url})")
