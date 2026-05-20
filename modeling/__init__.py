"""
Throughput Prediction Framework

Providers : starlink | oneweb | quectel
Entry point: run.py

Quick import examples
---------------------
    from dataset import DataConfig, build_dataloaders
    from model   import ThroughputSeq2Seq
    from models  import ModelRegistry
    from evaluation import evaluate_predictions
"""

from .dataset import (
    DataConfig,
    build_dataloaders,
    build_experiment_datasets,
    ThroughputFileDataset,
    ExperimentDataset,
    PROVIDERS,
    PROVIDER_STARLINK,
    PROVIDER_ONEWEB,
    PROVIDER_QUECTEL,
    ALL_FEATURES,
    DEFAULT_FEATURES,
    ALL_TARGETS,
    TARGET_ALIASES,
    DEFAULT_TARGET,
    FEATURE_GROUPS,
    ALL_FEATURE_NAMES,
)
from .model  import ThroughputSeq2Seq
from .models import ModelRegistry

__all__ = [
    # Dataset
    "DataConfig",
    "build_dataloaders",
    "build_experiment_datasets",
    "ThroughputFileDataset",
    "ExperimentDataset",
    # Provider constants
    "PROVIDERS",
    "PROVIDER_STARLINK",
    "PROVIDER_ONEWEB",
    "PROVIDER_QUECTEL",
    # Feature / target registries
    "ALL_FEATURES",
    "DEFAULT_FEATURES",
    "ALL_TARGETS",
    "TARGET_ALIASES",
    "DEFAULT_TARGET",
    "FEATURE_GROUPS",
    "ALL_FEATURE_NAMES",
    # Model
    "ThroughputSeq2Seq",
    "ModelRegistry",
]