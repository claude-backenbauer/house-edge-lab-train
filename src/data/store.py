"""Dataset store.

A dead-simple, dependency-free database: one JSON object per line in a ``.jsonl``
file. Append examples as you collect them; read them all back for training.
This is deliberately boring and portable -- your buddy can copy the file to the
GPU box and load it with the same class.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

from src.data.schema import TrainingExample


class DatasetStore:
    def __init__(self, path: str) -> None:
        self.path = path

    # ------------------------------------------------------------------ #
    def append(self, example: TrainingExample) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(example.to_dict()) + "\n")

    def extend(self, examples: Iterable[TrainingExample]) -> int:
        n = 0
        for ex in examples:
            self.append(ex)
            n += 1
        return n

    def __iter__(self) -> Iterator[TrainingExample]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield TrainingExample.from_dict(json.loads(line))

    def all(self) -> list[TrainingExample]:
        return list(self)

    def labeled(self) -> list[TrainingExample]:
        """Only examples with a known outcome (usable for supervised training)."""
        return [ex for ex in self if ex.is_labeled()]

    def __len__(self) -> int:
        return sum(1 for _ in self)
