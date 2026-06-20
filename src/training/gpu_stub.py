"""GPU training stub.

This module is a *placeholder*. It trains nothing and loads no model. It
documents how a future learned forecaster would be trained on GPU, what data it
would consume, and what it would predict -- so the rest of the lab can be wired
up against a stable interface before any real model exists.

Calling :func:`train` raises ``NotImplementedError`` on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# What the future training dataset should contain (one row per resolved market).
TRAINING_FEATURES: list[str] = [
    "market_question_text",  # raw text -> tokenised / embedded
    "category",
    "close_time",
    "price_time_series",  # sequence of (timestamp, price-per-outcome)
    "volume",
    "liquidity",
    "resolution_source",
    "final_outcome",  # the label for probability calibration
    "trader_flow_features",  # optional: net informed flow, order imbalance, etc.
]

# What the future model should predict.
TRAINING_TARGETS: list[str] = [
    "calibrated_event_probability",  # well-calibrated P(outcome)
    "expected_volume",
    "adverse_selection_risk",
    "expected_profit_by_fee_liquidity",  # surface over (creator_fee, lp_fee, liq)
]


@dataclass
class TrainingPlan:
    """Static description of the (future) training setup."""

    framework: str = "pytorch"
    accelerator: str = "cuda"  # or "mps" / "rocm" depending on hardware
    architecture: str = (
        "text encoder (small transformer / frozen sentence-embedding) + "
        "temporal encoder (GRU/TCN over the price-volume series) + "
        "tabular MLP head; multi-task outputs with a calibration layer"
    )
    losses: list[str] = field(
        default_factory=lambda: [
            "binary/multiclass cross-entropy for outcome probability",
            "temperature scaling / focal calibration term",
            "MSE (log-space) for expected volume",
            "quantile/pinball loss for adverse-selection risk",
            "MSE for the profit-vs-(fee,liquidity) surface",
        ]
    )
    data_splits: tuple[str, ...] = (
        "time-based train/val/test split to avoid look-ahead leakage",
    )
    notes: str = (
        "v1 ships NO trained model. This module only documents the intended "
        "pipeline. Live platform/API data collection is also out of scope."
    )


def describe_training_plan() -> dict:
    """Return a serialisable description of the future training pipeline."""
    plan = TrainingPlan()
    return {
        "status": "not_implemented",
        "framework": plan.framework,
        "accelerator": plan.accelerator,
        "architecture": plan.architecture,
        "losses": plan.losses,
        "data_splits": list(plan.data_splits),
        "dataset_features": list(TRAINING_FEATURES),
        "prediction_targets": list(TRAINING_TARGETS),
        "notes": plan.notes,
    }


def train(*args, **kwargs):  # noqa: D401 - intentional stub
    """Intentionally unimplemented. See :func:`describe_training_plan`."""
    raise NotImplementedError(
        "GPU training is not implemented in v1. house-edge-lab is a simulation "
        "tool. See describe_training_plan() for the intended future pipeline."
    )
