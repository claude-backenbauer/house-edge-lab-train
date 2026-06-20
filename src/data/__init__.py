"""Data layer: training-example schema, a simple store, and data sources."""

from src.data.schema import TrainingExample, PricePoint
from src.data.store import DatasetStore
from src.data.sources import DATA_SOURCES, recommended_sources, print_sources
from src.data.collectors import (
    COLLECTORS,
    ManifoldCollector,
    PolymarketCollector,
)

__all__ = [
    "TrainingExample",
    "PricePoint",
    "DatasetStore",
    "DATA_SOURCES",
    "recommended_sources",
    "print_sources",
    "COLLECTORS",
    "ManifoldCollector",
    "PolymarketCollector",
]
