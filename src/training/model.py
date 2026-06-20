"""The model definition (PyTorch).

A small multi-task network your buddy trains on GPU. It's intentionally modest
for v1 -- it learns from the simple numeric features in ``featurize.py``. The
architecture leaves room to bolt on a real text encoder later (the question
text is the biggest untapped signal).

PyTorch is imported lazily so the rest of house-edge-lab runs fine on machines
without it (like this one). On the GPU box, ``pip install torch`` first.
"""

from __future__ import annotations


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "PyTorch is required to build/train the model. Install it on the "
            "GPU machine with `pip install torch`. (The rest of house-edge-lab "
            "works without it.)"
        ) from e
    return torch, nn


def build_model(n_features: int, hidden: int = 64):
    """Create the multi-task model.

    Heads:
      * outcome_prob -- P(first outcome wins)        [calibrated probability]
      * volume       -- log expected volume          [regression]
      * adverse_risk -- adverse-selection risk 0..1  [regression]

    Returns an ``nn.Module``. Requires PyTorch.
    """
    torch, nn = _require_torch()

    class HouseEdgeNet(nn.Module):
        def __init__(self, n_in: int, h: int):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_in, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
            )
            self.head_prob = nn.Linear(h, 1)
            self.head_volume = nn.Linear(h, 1)
            self.head_adverse = nn.Linear(h, 1)
            # Temperature for calibration (learned).
            self.log_temp = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            z = self.trunk(x)
            logit = self.head_prob(z) / self.log_temp.exp()
            return {
                "outcome_logit": logit.squeeze(-1),
                "outcome_prob": torch.sigmoid(logit).squeeze(-1),
                "log_volume": self.head_volume(z).squeeze(-1),
                "adverse_risk": torch.sigmoid(self.head_adverse(z)).squeeze(-1),
            }

    return HouseEdgeNet(n_features, hidden)
