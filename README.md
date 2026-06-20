# house-edge-lab — training bundle

Public, training-only slice of **house-edge-lab** (a prediction-market design
*simulation* tool). Contains just the code, the quality-swept market dataset,
and a ready-to-run notebook — enough to train and evaluate the model.

> **Simulation only.** Nothing here places real bets, trades, or touches any
> real-money system.

## Run it on molab (free GPU, no install)

1. Open <https://molab.marimo.io> and sign in.
2. **New notebook → Mirror from GitHub**, paste:
   `https://github.com/claude-backenbauer/house-edge-lab-train/blob/main/notebooks/train_molab.py`
3. Click **Run all**. The notebook fetches this code automatically, trains the
   model, and shows the scoreboard — with plain-English notes on each step.

Or run anywhere with Python + PyTorch:

```bash
pip install torch
python -m src.training.train --data data/markets.jsonl --epochs 50
```

## What's inside

| Path | What |
|---|---|
| `src/` | the lab: validation, economics, market-maker sims, forecasting, telemetry, data tools |
| `data/markets.jsonl` | quality-swept resolved markets (Manifold + Polymarket), each with a source stamp |
| `notebooks/train_molab.py` | the ready-to-run molab notebook |
| `tests/` | the test suite (`python -m unittest discover -s tests`) |

The data was collected read-only from public APIs and filtered for reliability
(real questions, enough participants, clean resolutions).
