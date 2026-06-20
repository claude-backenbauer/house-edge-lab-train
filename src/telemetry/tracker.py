"""Telemetry tracker.

The scoreboard for "how good is our predictor, and is it getting better?"

Workflow:
    1. The model makes a prediction  -> ``record_prediction(...)``.
    2. The real event resolves       -> ``resolve(market_id, winner)``.
    3. Read the scoreboard           -> ``summary()`` / ``report()``.

Everything is appended to a plain-text log file (one JSON object per line) so
nothing is ever lost and you can replay history. The metrics tracked:

    accuracy        -- how often the most-likely outcome actually won
    brier_score     -- prediction error (LOWER is better; 0 = perfect)
    log_loss        -- punishes confident wrong calls (LOWER is better)
    calibration     -- when we say "70%", does it happen ~70% of the time?
    avg_confidence  -- how sure the model claims to be
    profit          -- if bet sizes were attached, did we make money?

These are exactly the numbers to watch as your buddy trains the model: if the
model is learning, brier_score and log_loss fall and calibration tightens.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from src.predictors.base import Prediction


@dataclass
class PredictionRecord:
    """One prediction, optionally resolved with the real outcome."""

    market_id: str
    model: str
    outcomes: list[str]
    probabilities: list[float]
    confidence: float
    created_at: str
    # filled in once the event resolves:
    resolved: bool = False
    winner_index: int | None = None
    winner: str | None = None
    resolved_at: str | None = None
    # optional economics, if you tracked a bet:
    stake: float = 0.0
    payout: float = 0.0

    # --- per-record metrics (only meaningful once resolved) ------------- #
    def brier(self) -> float | None:
        if not self.resolved or self.winner_index is None:
            return None
        return sum(
            (p - (1.0 if i == self.winner_index else 0.0)) ** 2
            for i, p in enumerate(self.probabilities)
        )

    def log_loss(self) -> float | None:
        if not self.resolved or self.winner_index is None:
            return None
        p = self.probabilities[self.winner_index]
        p = min(max(p, 1e-9), 1 - 1e-9)
        return -math.log(p)

    def correct(self) -> bool | None:
        if not self.resolved or self.winner_index is None:
            return None
        top = max(range(len(self.probabilities)), key=lambda j: self.probabilities[j])
        return top == self.winner_index

    def profit(self) -> float:
        return self.payout - self.stake


@dataclass
class TelemetrySummary:
    model: str
    n_predictions: int
    n_resolved: int
    accuracy: float | None
    brier_score: float | None
    log_loss: float | None
    avg_confidence: float | None
    calibration_error: float | None  # mean gap between confidence and reality
    total_profit: float
    roi: float | None
    calibration_bins: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """A short, human-readable scoreboard."""
        def f(x, pct=False):
            if x is None:
                return "n/a (no resolved events yet)"
            return f"{x:.1%}" if pct else f"{x:.4f}"

        lines = [
            f"Model: {self.model}",
            f"  predictions made ......... {self.n_predictions}",
            f"  events resolved .......... {self.n_resolved}",
            f"  accuracy (picked winner) . {f(self.accuracy, pct=True)}",
            f"  brier score (lower=better) {f(self.brier_score)}",
            f"  log loss (lower=better) .. {f(self.log_loss)}",
            f"  avg confidence ........... {f(self.avg_confidence, pct=True)}",
            f"  calibration error ........ {f(self.calibration_error, pct=True)}",
            f"  total profit ............. {self.total_profit:,.2f}",
            f"  return on stake .......... {f(self.roi, pct=True)}",
        ]
        if self.calibration_bins:
            lines.append("  calibration (said -> actually happened):")
            for b in self.calibration_bins:
                if b["n"]:
                    lines.append(
                        f"    {b['range']}: said ~{b['said']:.0%}, "
                        f"happened {b['actual']:.0%}  (n={b['n']})"
                    )
        return "\n".join(lines)


class TelemetryTracker:
    """Records predictions and outcomes; computes evolving metrics.

    Parameters
    ----------
    log_path
        Optional JSONL file to append every event to (durable history).
    model
        Name/tag of the model being tracked (defaults to "model").
    """

    def __init__(self, log_path: str | None = None, model: str = "model") -> None:
        self.model = model
        self.log_path = log_path
        self._records: dict[str, PredictionRecord] = {}
        if log_path and os.path.exists(log_path):
            self._load(log_path)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_prediction(
        self,
        prediction: Prediction,
        stake: float = 0.0,
    ) -> PredictionRecord:
        rec = PredictionRecord(
            market_id=prediction.market_id,
            model=prediction.source or self.model,
            outcomes=list(prediction.outcomes),
            probabilities=list(prediction.probabilities),
            confidence=prediction.confidence,
            created_at=prediction.created_at,
            stake=stake,
        )
        self._records[rec.market_id] = rec
        self._append({"event": "prediction", **rec_to_log(rec)})
        return rec

    def resolve(
        self,
        market_id: str,
        winner: str | int,
        payout: float | None = None,
    ) -> PredictionRecord | None:
        """Record the real outcome for a previously-predicted market.

        ``winner`` may be the outcome label or its index. If ``payout`` is given
        (and a stake was recorded), profit is tracked too.
        """
        rec = self._records.get(market_id)
        if rec is None:
            return None
        if isinstance(winner, int):
            idx = winner
        else:
            idx = next(
                (i for i, o in enumerate(rec.outcomes) if o.lower() == winner.lower()),
                None,
            )
        if idx is None:
            raise ValueError(f"winner '{winner}' not in outcomes {rec.outcomes}")
        rec.resolved = True
        rec.winner_index = idx
        rec.winner = rec.outcomes[idx]
        rec.resolved_at = datetime.now(timezone.utc).isoformat()
        if payout is not None:
            rec.payout = payout
        self._append({"event": "resolution", **rec_to_log(rec)})
        return rec

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def summary(self, n_bins: int = 5) -> TelemetrySummary:
        recs = list(self._records.values())
        resolved = [r for r in recs if r.resolved]

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None

        accuracy = mean([1.0 if r.correct() else 0.0 for r in resolved]) if resolved else None
        brier = mean([r.brier() for r in resolved])
        logloss = mean([r.log_loss() for r in resolved])
        avg_conf = mean([r.confidence for r in recs]) if recs else None
        total_profit = sum(r.profit() for r in resolved)
        total_stake = sum(r.stake for r in resolved)
        roi = (total_profit / total_stake) if total_stake > 0 else None

        bins, cal_err = self._calibration(resolved, n_bins)

        return TelemetrySummary(
            model=self.model,
            n_predictions=len(recs),
            n_resolved=len(resolved),
            accuracy=accuracy,
            brier_score=brier,
            log_loss=logloss,
            avg_confidence=avg_conf,
            calibration_error=cal_err,
            total_profit=total_profit,
            roi=roi,
            calibration_bins=bins,
        )

    def report(self) -> str:
        return self.summary().render()

    def _calibration(self, resolved: list[PredictionRecord], n_bins: int):
        """Reliability bins on the model's confidence in its top pick."""
        bins = [
            {"range": f"{i/n_bins:.0%}-{(i+1)/n_bins:.0%}", "said": 0.0,
             "actual": 0.0, "n": 0, "_s": 0.0, "_a": 0.0}
            for i in range(n_bins)
        ]
        for r in resolved:
            top_i = max(range(len(r.probabilities)),
                        key=lambda j: r.probabilities[j])
            p = r.probabilities[top_i]
            b = min(n_bins - 1, int(p * n_bins))
            bins[b]["n"] += 1
            bins[b]["_s"] += p
            bins[b]["_a"] += 1.0 if top_i == r.winner_index else 0.0
        gaps = []
        for b in bins:
            if b["n"]:
                b["said"] = b["_s"] / b["n"]
                b["actual"] = b["_a"] / b["n"]
                gaps.append(abs(b["said"] - b["actual"]))
            b.pop("_s"); b.pop("_a")
        cal_err = sum(gaps) / len(gaps) if gaps else None
        return bins, cal_err

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _append(self, obj: dict) -> None:
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                mid = obj.get("market_id")
                if not mid:
                    continue
                rec = self._records.get(mid) or PredictionRecord(
                    market_id=mid,
                    model=obj.get("model", self.model),
                    outcomes=obj.get("outcomes", []),
                    probabilities=obj.get("probabilities", []),
                    confidence=obj.get("confidence", 0.5),
                    created_at=obj.get("created_at", ""),
                )
                for k in ("resolved", "winner_index", "winner", "resolved_at",
                          "stake", "payout"):
                    if k in obj and obj[k] is not None:
                        setattr(rec, k, obj[k])
                self._records[mid] = rec


def rec_to_log(rec: PredictionRecord) -> dict:
    d = asdict(rec)
    return d
