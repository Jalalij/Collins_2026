"""
Throughput Prediction Framework
===============================

This package provides the main components used by the throughput and
network-QoS prediction framework.

Supported providers
-------------------
    - Starlink
    - OneWeb
    - Quectel

Main entry point
----------------
    run.py

Typical usage
-------------
Dataset utilities:

    from dataset import DataConfig, build_dataloaders

Seq2Seq model:

    from model import ThroughputSeq2Seq

Model registry:

    from models import ModelRegistry

Evaluation utilities:

    from evaluation import evaluate_predictions

This __init__.py file exposes the most commonly used dataset, feature,
target, provider, and model objects at the package level so that other
modules can import them directly without referencing their internal files.
"""


# ============================================================================
# Dataset configuration and data-loading utilities
# ============================================================================
#
# These imports provide:
#
#   - DataConfig:
#       Central configuration object describing provider, experiment type,
#       selected features, targets, input/output lengths, normalization, etc.
#
#   - build_dataloaders:
#       Creates the PyTorch train / validation / test DataLoaders.
#
#   - build_experiment_datasets:
#       Builds datasets for a configured experiment before wrapping them
#       inside DataLoaders.
#
#   - ThroughputFileDataset:
#       Dataset representing samples generated from an individual data file.
#
#   - ExperimentDataset:
#       Higher-level dataset abstraction used for complete experiments.
#
# The remaining objects define provider names, valid features, valid targets,
# aliases, and feature groups used throughout the framework.
# ============================================================================

from .dataset import (
    DataConfig,
    build_dataloaders,
    build_experiment_datasets,
    ThroughputFileDataset,
    ExperimentDataset,

    # ------------------------------------------------------------------------
    # Supported network/provider constants
    # ------------------------------------------------------------------------
    PROVIDERS,
    PROVIDER_STARLINK,
    PROVIDER_ONEWEB,
    PROVIDER_QUECTEL,

    # ------------------------------------------------------------------------
    # Feature and target registries
    # ------------------------------------------------------------------------
    #
    # ALL_FEATURES:
    #     Complete set of supported input features.
    #
    # DEFAULT_FEATURES:
    #     Default input feature selection used when no explicit feature list
    #     is provided.
    #
    # ALL_TARGETS:
    #     Complete set of prediction targets supported by the framework.
    #
    # TARGET_ALIASES:
    #     Maps convenient short names such as "rtt", "jitter", or "loss"
    #     to their corresponding dataset column names.
    #
    # DEFAULT_TARGET:
    #     Default prediction target when none is explicitly specified.
    #
    # FEATURE_GROUPS:
    #     Logical groups of related features used for experiment selection
    #     and ablation studies.
    #
    # ALL_FEATURE_NAMES:
    #     Flattened / consolidated collection of valid feature names.
    # ------------------------------------------------------------------------
    ALL_FEATURES,
    DEFAULT_FEATURES,
    ALL_TARGETS,
    TARGET_ALIASES,
    DEFAULT_TARGET,
    FEATURE_GROUPS,
    ALL_FEATURE_NAMES,
)


# ============================================================================
# Seq2Seq model
# ============================================================================
#
# ThroughputSeq2Seq is the GRU-based sequence-to-sequence forecasting model
# used for multi-step prediction of network metrics such as:
#
#   - RTT
#   - jitter
#   - packet loss
#   - throughput
#
# Importing it here allows users to write:
#
#     from package_name import ThroughputSeq2Seq
#
# instead of importing it from its internal module directly.
# ============================================================================

from .model import ThroughputSeq2Seq


# ============================================================================
# Model registry
# ============================================================================
#
# ModelRegistry provides a common mechanism for selecting and constructing
# the supported forecasting models.
#
# Depending on the framework configuration, this may include models such as:
#
#   - Seq2Seq GRU
#   - LSTM
#   - Transformer
#   - other registered baselines
#
# This allows run.py to select models dynamically using command-line names
# instead of hardcoding model construction logic.
# ============================================================================

from .models import ModelRegistry


# ============================================================================
# Public package API
# ============================================================================
#
# __all__ defines which objects are considered part of the package's public
# interface.
#
# For example:
#
#     from package_name import *
#
# will expose only the objects listed below.
#
# Keeping this list explicit also makes the intended external API easier to
# understand and prevents internal helper objects from being exposed
# accidentally.
# ============================================================================

__all__ = [

    # ------------------------------------------------------------------------
    # Dataset configuration and loading
    # ------------------------------------------------------------------------
    "DataConfig",
    "build_dataloaders",
    "build_experiment_datasets",
    "ThroughputFileDataset",
    "ExperimentDataset",

    # ------------------------------------------------------------------------
    # Supported providers
    # ------------------------------------------------------------------------
    "PROVIDERS",
    "PROVIDER_STARLINK",
    "PROVIDER_ONEWEB",
    "PROVIDER_QUECTEL",

    # ------------------------------------------------------------------------
    # Feature and target definitions
    # ------------------------------------------------------------------------
    "ALL_FEATURES",
    "DEFAULT_FEATURES",
    "ALL_TARGETS",
    "TARGET_ALIASES",
    "DEFAULT_TARGET",
    "FEATURE_GROUPS",
    "ALL_FEATURE_NAMES",

    # ------------------------------------------------------------------------
    # Forecasting models
    # ------------------------------------------------------------------------
    "ThroughputSeq2Seq",
    "ModelRegistry",
]