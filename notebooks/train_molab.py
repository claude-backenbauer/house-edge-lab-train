"""house-edge-lab — ready-to-run molab training notebook.

HOW TO USE (no coding needed):
  1. Open https://molab.marimo.io in your browser and sign in.
  2. Upload these together: this file, the `src/` folder, and `data/markets.jsonl`.
  3. Open this notebook and click "Run all".
  4. Watch the model train and the scoreboard appear.

This is a marimo notebook (it runs top-to-bottom). Each block has a plain-English
note above it. Nothing here touches real money.
"""

import marimo

__generated_with = "0.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # One-time setup: make sure the project code is present, then import it.
    import json
    import os
    import subprocess
    import sys

    import marimo as mo

    # If the code isn't already here (e.g. this notebook was mirrored on its
    # own), fetch the public training repo so `src/` and the dataset exist.
    _REPO = "https://github.com/claude-backenbauer/house-edge-lab-train.git"
    _DIR = "house-edge-lab-train"
    if not os.path.exists("src") and not os.path.exists(os.path.join(_DIR, "src")):
        subprocess.run(["git", "clone", "--depth", "1", _REPO], check=False)
    if os.path.exists(os.path.join(_DIR, "src")):
        os.chdir(_DIR)  # so data/markets.jsonl resolves
    if "." not in sys.path:
        sys.path.insert(0, ".")

    import torch

    from src.data.store import DatasetStore
    from src.training.featurize import build_xy
    from src.training.train import train

    return DatasetStore, build_xy, json, mo, os, torch, train


@app.cell
def _(mo):
    mo.md(
        """
        # Training the house-edge model

        This notebook trains a small model to predict market outcomes, then
        shows how well it did. Just run each block top to bottom.
        """
    )
    return


@app.cell
def _(mo, torch):
    # Step 1 — Is the free GPU switched on?
    gpu = torch.cuda.is_available()
    mo.md(
        f"**Step 1 — GPU check:** GPU available = **{gpu}**. "
        + ("✅ Great, the cloud GPU is on." if gpu
           else "ℹ️ No GPU detected — that's fine, this small model also trains "
                "on CPU. (Toggle the GPU via the 'specs' button to use it.)")
    )
    return (gpu,)


@app.cell
def _(DatasetStore, build_xy, mo):
    # Step 2 — Load the data we collected (resolved, quality-swept markets).
    store = DatasetStore("data/markets.jsonl")
    labeled = store.labeled()
    X, y = build_xy(labeled)
    yes = sum(1 for v in y if v == 1.0)
    mo.md(
        f"**Step 2 — Data:** {len(labeled)} reliable markets loaded "
        f"({yes} YES / {len(y) - yes} NO). Each row is a real, resolved market "
        f"with its history and a checked source."
    )
    return labeled, store, X, y


@app.cell
def _(mo, train):
    # Step 3 — Train. Watch the numbers improve (loss down, accuracy up).
    mo.md("**Step 3 — Training** (this runs the learning; see live numbers below):")
    model = train("data/markets.jsonl", epochs=50,
                  telemetry_path="reports/training_telemetry.jsonl")
    return (model,)


@app.cell
def _(json, mo, os):
    # Step 4 — Show the final scoreboard from the telemetry log.
    path = "reports/training_telemetry.jsonl"
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    if rows:
        first, last = rows[0], rows[-1]
        msg = (
            "**Step 4 — Results** (start → end):\n\n"
            f"- log-loss (lower better): {first['val_log_loss']:.3f} → "
            f"**{last['val_log_loss']:.3f}**\n"
            f"- brier (lower better): {first['val_brier']:.3f} → "
            f"**{last['val_brier']:.3f}**\n"
            f"- accuracy (higher better): {first['val_accuracy']:.0%} → "
            f"**{last['val_accuracy']:.0%}**\n\n"
            "If the end numbers beat the start, the model learned something. "
            "With a small/early dataset, don't expect miracles yet — this proves "
            "the whole pipeline runs end-to-end."
        )
    else:
        msg = "**Step 4 — Results:** no telemetry found (did Step 3 run?)."
    mo.md(msg)
    return (rows,)


if __name__ == "__main__":
    app.run()
