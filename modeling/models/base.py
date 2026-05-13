"""
models/base.py - BaseTrainer contract and central ModelRegistry.

Adding a new model
------------------
1. Create models/my_model.py with MyTrainer(BaseTrainer).
2. Implement train_and_evaluate() per the contract below.
3. Register at the bottom of this file.

Trainer contract
----------------
train_and_evaluate(train_set, test_set, results_dir, config) -> dict

    train_set / test_set : ThroughputFileDataset or ExperimentDataset
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
            "y_true_dict" : {target_name: np.ndarray},   # shape (N,) per target×horizon key
            "y_pred_dict" : {target_name: np.ndarray},
            "metadata"    : dict,
        }
    Keys in y_true_dict / y_pred_dict can be:
      - horizon labels  "t+1s", "t+2s", ...   (Seq2Seq single-target)
      - "var_0", "var_1", ...                  (legacy next-step trainers)
      - "dl_t+1s", "rtt_t+2s", ...            (multi-target horizon)
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @staticmethod
    @abstractmethod
    def train_and_evaluate(train_set, test_set, results_dir: str, config: dict) -> dict:
        raise NotImplementedError


class _ModelRegistry:
    def __init__(self):
        self._r: dict[str, dict] = {}

    def register(self, key, trainer_cls, description="", default_config=None):
        self._r[key] = {
            "cls":            trainer_cls,
            "description":    description,
            "default_config": default_config or {},
        }

    def get(self, key):
        if key not in self._r:
            raise KeyError(f"Unknown model '{key}'. Available: {list(self._r)}")
        return self._r[key]

    def list_models(self):
        return [{"key": k, "description": v["description"],
                 "default_config": v["default_config"]} for k, v in self._r.items()]

    def keys(self):
        return list(self._r.keys())

    def __contains__(self, key):
        return key in self._r


ModelRegistry = _ModelRegistry()


def _register_builtins():
    from .seq2seq            import Seq2SeqTrainer
    from .lstm_model         import LSTMTrainer
    from .transformer_model  import TransformerTrainer
    from .gaussian_process_model import GaussianProcessTrainer
    from .autoencoder_models import PredictiveAutoencoderTrainer

    ModelRegistry.register(
        "seq2seq", Seq2SeqTrainer,
        description="GRU Seq2Seq with attention + periodic time embedding (primary model)",
        default_config={"hidden_size": 128, "embed_dim": 16, "dropout": 0.3,
                        "epochs": 100, "lr": 1e-3, "batch_size": 512},
    )
    ModelRegistry.register(
        "lstm", LSTMTrainer,
        description="2-layer LSTM next-step forecaster",
        default_config={"hidden_size": 64, "num_layers": 2, "dropout": 0.2,
                        "epochs": 100, "lr": 1e-3, "batch_size": 64},
    )
    ModelRegistry.register(
        "transformer", TransformerTrainer,
        description="Encoder-only Transformer next-step forecaster",
        default_config={"d_model": 64, "nhead": 4, "num_layers": 2,
                        "dim_feedforward": 256, "dropout": 0.1,
                        "epochs": 100, "lr": 1e-3, "batch_size": 64},
    )
    ModelRegistry.register(
        "gaussian_process", GaussianProcessTrainer,
        description="Gaussian Process regressor (sklearn) — small datasets only",
        default_config={"train_samples": 5000, "test_samples": 1000},
    )
    ModelRegistry.register(
        "autoencoder", PredictiveAutoencoderTrainer,
        description="Predictive autoencoder (standard | sparse | variational)",
        default_config={"variant": "standard", "epochs": 100, "lr": 1e-3, "batch_size": 64},
    )


_register_builtins()
