"""Model training.

  * ``gpu_stub``  -- the documented plan / placeholder
  * ``featurize`` -- turn examples into numbers (no torch needed)
  * ``model``     -- the PyTorch model definition (needs torch)
  * ``train``     -- the GPU training loop with telemetry (needs torch)
"""

from src.training.gpu_stub import (
    TrainingPlan,
    describe_training_plan,
    TRAINING_FEATURES,
    TRAINING_TARGETS,
)
from src.training.featurize import FEATURE_NAMES, featurize, build_xy, label_yes

__all__ = [
    "TrainingPlan",
    "describe_training_plan",
    "TRAINING_FEATURES",
    "TRAINING_TARGETS",
    "FEATURE_NAMES",
    "featurize",
    "build_xy",
    "label_yes",
]
