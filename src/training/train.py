"""Training loop (PyTorch) -- runs on your buddy's GPU.

Reads the dataset built by ``DatasetStore``, trains the model from
``model.py``, and -- importantly for you -- logs telemetry every epoch so you
can watch the model learn (loss falling, accuracy rising, calibration
tightening). Telemetry is written to a JSONL file you can chart later.

PyTorch is required only here. On the GPU box:

    pip install torch
    python -m src.training.train --data data/markets.jsonl --epochs 50

This module has no effect on the rest of the lab if torch is absent.
"""

from __future__ import annotations

import argparse
import json
import os

from src.data.store import DatasetStore
from src.training.featurize import FEATURE_NAMES, build_xy
from src.training.model import build_model, _require_torch


def train(
    data_path: str,
    epochs: int = 50,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    telemetry_path: str = "reports/training_telemetry.jsonl",
    seed: int = 42,
):
    """Train the model and stream per-epoch telemetry. Requires PyTorch."""
    torch, nn = _require_torch()
    torch.manual_seed(seed)

    store = DatasetStore(data_path)
    examples = store.labeled()
    X, y = build_xy(examples)
    if len(X) < 8:
        raise SystemExit(
            f"Only {len(X)} labeled examples found in {data_path}. Collect more "
            "resolved markets first (see `python -m src.main sources`)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    n = len(X)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = build_model(len(FEATURE_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
    log = open(telemetry_path, "a", encoding="utf-8")
    print(f"Training on {device}: {len(tr_idx)} train / {len(val_idx)} val")

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            out = model(X_t[tr_idx])
            loss = bce(out["outcome_logit"], y_t[tr_idx])
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                vout = model(X_t[val_idx])
                vprob = vout["outcome_prob"].clamp(1e-6, 1 - 1e-6)
                vy = y_t[val_idx]
                val_loss = nn.functional.binary_cross_entropy(vprob, vy).item()
                brier = ((vprob - vy) ** 2).mean().item()
                acc = (((vprob > 0.5).float() == vy).float().mean().item())

            row = {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_log_loss": val_loss,
                "val_brier": brier,
                "val_accuracy": acc,
                "temperature": float(model.log_temp.exp().item()),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                print(
                    f"epoch {epoch:4d}  train_loss={loss.item():.4f}  "
                    f"val_log_loss={val_loss:.4f}  brier={brier:.4f}  "
                    f"acc={acc:.1%}"
                )
    finally:
        log.close()

    print(f"Done. Telemetry -> {telemetry_path}")
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the house-edge model (GPU).")
    p.add_argument("--data", required=True, help="path to dataset .jsonl")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--telemetry", default="reports/training_telemetry.jsonl")
    args = p.parse_args(argv)
    train(args.data, epochs=args.epochs, lr=args.lr,
          telemetry_path=args.telemetry)


if __name__ == "__main__":
    main()
