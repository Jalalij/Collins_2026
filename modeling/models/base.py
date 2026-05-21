"""
models/base.py - BaseTrainer contract and central ModelRegistry.

Adding a new model
------------------
1. Create models/my_model.py with MyTrainer(BaseTrainer).
2. Implement train_and_evaluate() per the contract below.
3. Import and register in _register_builtins() at the bottom of this file.
4. Add to models/__init__.py imports and __all__.

Trainer contract
----------------
train_and_evaluate(train_set, test_set, results_dir, config, val_set=None) -> dict

    train_set / test_set / val_set :
        ThroughputFileDataset or ExperimentDataset (val_set may be None)
        - __len__, __getitem__ -> (X [T,F], y [H] or [H,num_targets])
        - .target_cols  : list[str]
        - .target_stats : {col: {mean,std}} or {col: {min,max}}
        - .normalise    : str
        - .inverse_transform(tensor, target_idx=0) -> tensor
        - .inverse_transform_all(tensor)           -> tensor  (multi-target)

    results_dir : str
    config      : dict  (model hyperparameters)

    Returns dict with at minimum:
        {
            "model_name"  : str,
            "y_true_dict" : {key: np.ndarray shape (N,)},
            "y_pred_dict" : {key: np.ndarray shape (N,)},
            "metadata"    : dict,
        }

    Keys in y_true_dict / y_pred_dict:
        "t+1s", "t+2s", ...          single-target Seq2Seq horizon labels
        "dl_t+1s", "rtt_t+2s", ...   multi-target Seq2Seq horizon labels
        "var_0", "var_1", ...         legacy next-step trainers
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @staticmethod
    @abstractmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        raise NotImplementedError


class _ModelRegistry:
    def __init__(self):
        self._r: dict[str, dict] = {}

    def register(self, key: str, trainer_cls, description: str = "",
                 default_config: dict | None = None) -> None:
        self._r[key] = {
            "cls":            trainer_cls,
            "description":    description,
            "default_config": default_config or {},
        }

    def get(self, key: str) -> dict:
        if key not in self._r:
            raise KeyError(f"Unknown model '{key}'. Available: {list(self._r)}")
        return self._r[key]

    def list_models(self) -> list[dict]:
        return [
            {"key": k, "description": v["description"],
             "default_config": v["default_config"]}
            for k, v in self._r.items()
        ]

    def keys(self) -> list[str]:
        return list(self._r.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._r


ModelRegistry = _ModelRegistry()


def _register_builtins() -> None:
    """
    Import all built-in trainers and register them with ModelRegistry.

    Called once from models/__init__.py (not at module import time) so that
    a broken trainer file doesn't prevent the rest of the package from loading.
    run.py also imports and calls this explicitly as a belt-and-suspenders guard.
    """
    from .seq2seq                import Seq2SeqTrainer
    from .lstm_model             import LSTMTrainer
    from .transformer_model      import TransformerTrainer
    from .gaussian_process_model import GaussianProcessTrainer
    from .autoencoder_models     import PredictiveAutoencoderTrainer
    from .starnet_models       import (
        SatelliteGRUTrainer, SatelliteAttentionTrainer, SatelliteFNNTrainer,
    )

    ModelRegistry.register(
        "seq2seq", Seq2SeqTrainer,
        description="GRU Seq2Seq with attention + periodic time embedding (primary model)",
        default_config={
            "hidden_size": 128, "embed_dim": 16, "dropout": 0.3,
            "epochs": 100, "lr": 1e-3, "batch_size": 512,
        },
    )
    ModelRegistry.register(
        "lstm", LSTMTrainer,
        description="2-layer LSTM next-step forecaster",
        default_config={
            "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
            "epochs": 100, "lr": 1e-3, "batch_size": 64,
        },
    )
    ModelRegistry.register(
        "transformer", TransformerTrainer,
        description="Encoder-only Transformer next-step forecaster",
        default_config={
            "d_model": 64, "nhead": 4, "num_layers": 2,
            "dim_feedforward": 256, "dropout": 0.1,
            "epochs": 100, "lr": 1e-3, "batch_size": 64,
        },
    )
    ModelRegistry.register(
        "gaussian_process", GaussianProcessTrainer,
        description="Gaussian Process regressor (sklearn) — small datasets only",
        default_config={"train_samples": 5000, "test_samples": 1000},
    )
    ModelRegistry.register(
        "autoencoder", PredictiveAutoencoderTrainer,
        description="Predictive autoencoder (standard | sparse | variational)",
        default_config={
            "variant": "standard", "epochs": 100, "lr": 1e-3, "batch_size": 64,
        },
    )
    ModelRegistry.register(
        "satellite_gru", SatelliteGRUTrainer,
        description="Satellite GRU Seq2Seq encoder-decoder",
        default_config={
            "hidden_size": 128, "num_layers": 2,
            "epochs": 100, "lr": 1e-3, "batch_size": 256, "patience": 15,
        },
    )
    ModelRegistry.register(
        "satellite_attention", SatelliteAttentionTrainer,
        description="Satellite GRU with Bahdanau attention",
        default_config={
            "hidden_size": 128, "num_layers": 2,
            "epochs": 100, "lr": 1e-3, "batch_size": 256, "patience": 15,
        },
    )
    ModelRegistry.register(
        "satellite_fnn", SatelliteFNNTrainer,
        description="Satellite feature-compress FNN (fastest)",
        default_config={
            "hidden_size": 128, "num_layers": 2,
            "epochs": 100, "lr": 1e-3, "batch_size": 256, "patience": 15,
        },
    )