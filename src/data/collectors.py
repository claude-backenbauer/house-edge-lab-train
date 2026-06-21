"""Read-only data collectors for resolved prediction markets.

Security posture (matches security-lab/AGENT_RULES.md):
  * READ-ONLY. Only HTTP GET. Never posts, trades, or authenticates.
  * NO installs. Uses the Python standard library (urllib) only.
  * Public endpoints only. No API keys, no logins.
  * Polite: identifies itself, times out, sleeps between calls, and is capped
    so it can't hammer a service or pull unbounded data.

These turn public, already-resolved markets into :class:`TrainingExample`
rows for our dataset. We never place orders or touch real money.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from src.data.quality import infer_topic, manifold_reliable, polymarket_reliable
from src.data.schema import PricePoint, TrainingExample

# Polymarket's own category strings are free-form; only trust these.
_KNOWN_TOPICS = {"politics", "economics", "crypto", "sports", "tech",
                 "science", "geopolitics"}

_USER_AGENT = "house-edge-lab/0.1 (research; read-only; contact: local user)"
_TIMEOUT = 20
_POLITE_DELAY = 0.25  # seconds between requests


def _get_json(url: str, params: dict | None = None, retries: int = 4):
    """Polite, read-only HTTP GET returning parsed JSON, with retries.

    Retries transient network errors with backoff so a long collection run
    isn't killed by a single blip. Raises only after all attempts fail.
    """
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    if not url.lower().startswith("https://"):
        raise ValueError("refusing non-HTTPS request")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
                data = resp.read().decode("utf-8")
            time.sleep(_POLITE_DELAY)
            return json.loads(data)
        except Exception as e:  # noqa: BLE001 - transient network/JSON errors
            last_err = e
            time.sleep(_POLITE_DELAY * (attempt + 1) * 2)
    raise last_err  # type: ignore[misc]


def _ms_to_iso(ms) -> str | None:
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except (ValueError, OSError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Manifold Markets  (public, no key)
# --------------------------------------------------------------------------- #

class ManifoldCollector:
    """Collect resolved BINARY markets from Manifold (api.manifold.markets)."""

    BASE = "https://api.manifold.markets/v0"
    name = "manifold"

    def collect(
        self,
        limit: int = 50,
        with_history: bool = False,
        max_scan: int = 6000,
        min_volume: float = 250.0,
        min_unique_traders: int = 15,
    ) -> list[TrainingExample]:
        """Return up to ``limit`` *reliable* resolved binary markets.

        Every candidate passes the quality sweep in ``src.data.quality`` before
        it's kept. Drop reasons are tallied in ``self.last_stats`` so the caller
        can show exactly what was scanned, kept, and weeded out.

        ``with_history=True`` additionally pulls each market's bet history to
        build a real price series (one extra request per kept market).
        """
        out: list[TrainingExample] = []
        scanned = 0
        drops: Counter = Counter()
        before: str | None = None

        while len(out) < limit and scanned < max_scan:
            try:
                page = _get_json(
                    f"{self.BASE}/markets",
                    {"limit": 100, "sort": "updated-time", "order": "desc",
                     **({"before": before} if before else {})},
                )
            except Exception:  # noqa: BLE001 - keep whatever we've gathered
                break
            if not page:
                break
            for m in page:
                scanned += 1
                keep, reason = manifold_reliable(
                    m, min_unique_traders=min_unique_traders,
                    min_volume=min_volume,
                )
                if not keep:
                    drops[reason] += 1
                    continue
                ex = self._to_example(m, with_history=with_history)
                if ex is None:
                    drops["mapping-failed"] += 1
                    continue
                out.append(ex)
                if len(out) >= limit:
                    break
            before = page[-1].get("id")
            if before is None:
                break

        self.last_stats = {
            "source": "manifold",
            "scanned": scanned,
            "kept": len(out),
            "dropped": dict(drops),
        }
        return out

    def _to_example(self, m: dict, with_history: bool) -> TrainingExample | None:
        if m.get("outcomeType") != "BINARY":
            return None
        if not m.get("isResolved"):
            return None
        resolution = m.get("resolution")
        if resolution not in ("YES", "NO"):
            return None  # skip MKT / CANCEL (no clean binary label)

        winner_idx = 0 if resolution == "YES" else 1
        prob = m.get("probability")
        if prob is None:
            prob = m.get("resolutionProbability", 0.5)

        series: list[PricePoint] = []
        if with_history:
            series = self._price_history(m.get("id"))
        if not series:
            t = _ms_to_iso(m.get("resolutionTime")) or _ms_to_iso(m.get("closeTime"))
            series = [PricePoint(t=t or "", prices=[prob, 1 - prob],
                                 volume=m.get("volume", 0.0) or 0.0)]

        return TrainingExample(
            market_id=str(m.get("id")),
            question=m.get("question", ""),
            description=m.get("textDescription", "") or "",
            category=infer_topic(m.get("question", "")),
            platform="manifold",
            resolution_source="manifold",
            outcomes=["YES", "NO"],
            close_time=_ms_to_iso(m.get("closeTime")),
            event_time=_ms_to_iso(m.get("resolutionTime")),
            price_series=series,
            final_volume=float(m.get("volume", 0.0) or 0.0),
            liquidity=float(sum((m.get("pool") or {}).values()) or 0.0),
            provenance={
                "source": "manifold",
                "url": f"https://manifold.markets/market/{m.get('slug', '')}",
                "unique_bettors": m.get("uniqueBettorCount"),
                "volume": float(m.get("volume", 0.0) or 0.0),
                "creator": m.get("creatorUsername"),
                "collected_via": "public read-only API",
            },
            final_outcome=resolution,
            final_outcome_index=winner_idx,
        )

    def _price_history(self, market_id, cap: int = 200) -> list[PricePoint]:
        if not market_id:
            return []
        try:
            bets = _get_json(f"{self.BASE}/bets",
                             {"contractId": market_id, "limit": cap})
        except Exception:
            return []
        pts = []
        for b in reversed(bets):  # API returns newest-first
            p = b.get("probAfter")
            if p is None:
                continue
            pts.append(PricePoint(t=_ms_to_iso(b.get("createdTime")) or "",
                                  prices=[p, 1 - p],
                                  volume=abs(b.get("amount", 0.0) or 0.0)))
        return pts


# --------------------------------------------------------------------------- #
# Polymarket  (public Gamma API, no key)
# --------------------------------------------------------------------------- #

class PolymarketCollector:
    """Collect closed/resolved markets from Polymarket's public Gamma API.

    Note: Polymarket's Gamma API does not expose a single 'winner' field, so we
    infer the winner from settled outcome prices (the outcome priced ~1.0).
    Full intraday price history needs the separate CLOB timeseries endpoint --
    a future add; v1 stores settled prices + volume.
    """

    BASE = "https://gamma-api.polymarket.com"
    name = "polymarket"

    def collect(self, limit: int = 50, max_scan: int = 800,
                min_volume: float = 1000.0) -> list[TrainingExample]:
        out: list[TrainingExample] = []
        offset = 0
        scanned = 0
        drops: Counter = Counter()
        while len(out) < limit and offset < max_scan:
            page = _get_json(
                f"{self.BASE}/markets",
                {"closed": "true", "limit": 100, "offset": offset,
                 "order": "volume", "ascending": "false"},
            )
            if not page:
                break
            for m in page:
                scanned += 1
                ex, reason = self._to_example(m, min_volume=min_volume)
                if ex is None:
                    drops[reason] += 1
                    continue
                out.append(ex)
                if len(out) >= limit:
                    break
            offset += 100
        self.last_stats = {
            "source": "polymarket", "scanned": scanned,
            "kept": len(out), "dropped": dict(drops),
        }
        return out

    @staticmethod
    def _category(m: dict) -> str:
        raw = (m.get("category") or "").strip().lower()
        if raw in _KNOWN_TOPICS:
            return raw
        return infer_topic(m.get("question", ""))

    def _to_example(self, m: dict, min_volume: float = 1000.0):
        """Return (TrainingExample|None, reason)."""
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = [float(p) for p in json.loads(m.get("outcomePrices", "[]"))]
        except (ValueError, TypeError):
            return None, "unparseable"

        vol = m.get("volumeNum")
        if vol is None:
            try:
                vol = float(m.get("volume", 0.0))
            except (ValueError, TypeError):
                vol = 0.0

        keep, reason = polymarket_reliable(
            outcomes, prices, float(vol or 0.0), m.get("question", ""),
            min_volume=min_volume,
        )
        if not keep:
            return None, reason

        winner_idx = next(i for i, p in enumerate(prices) if p >= 0.99)

        return TrainingExample(
            market_id=str(m.get("id") or m.get("conditionId") or m.get("slug")),
            question=m.get("question", ""),
            description=(m.get("description", "") or "")[:1000],
            category=self._category(m),
            platform="polymarket",
            resolution_source="polymarket",
            outcomes=outcomes,
            close_time=m.get("closedTime") or m.get("endDate"),
            event_time=m.get("closedTime") or m.get("endDate"),
            price_series=[],  # history via CLOB API is a future add
            final_volume=float(vol or 0.0),
            liquidity=float(m.get("liquidityNum", 0.0) or 0.0),
            provenance={
                "source": "polymarket",
                "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                "volume": float(vol or 0.0),
                "collected_via": "public read-only Gamma API",
            },
            final_outcome=outcomes[winner_idx],
            final_outcome_index=winner_idx,
        ), "ok"


COLLECTORS = {
    "manifold": ManifoldCollector,
    "polymarket": PolymarketCollector,
}
