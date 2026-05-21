"""
models/ - All model trainer implementations.
Registry-based — use ModelRegistry to discover and run models.
See models/base.py for the BaseTrainer contract.
"""

from .base import BaseTrainer, ModelRegistry, _register_builtins
_register_builtins()

from .seq2seq            import Seq2SeqTrainer
from .lstm_model         import LSTMTrainer, SimpleLSTM
from .transformer_model  import TransformerTrainer, SimpleTransformer
from .gaussian_process_model import GaussianProcessTrainer
from .starnet_models import (
    SatelliteGRUTrainer, SatelliteAttentionTrainer, SatelliteFNNTrainer,
    SatelliteGRU_seq2seq, SatelliteGRU_attention, Satellite_FNN,
)
from .autoencoder_models import (
    PredictiveAutoencoderTrainer,
    PredictiveAutoencoder,
    SparseAutoencoder,
    VariationalAutoencoder,
)

__all__ = [
    "BaseTrainer", "ModelRegistry",
    "Seq2SeqTrainer",
    "LSTMTrainer", "SimpleLSTM",
    "TransformerTrainer", "SimpleTransformer",
    "GaussianProcessTrainer",
    "PredictiveAutoencoderTrainer",
    "PredictiveAutoencoder", "SparseAutoencoder", "VariationalAutoencoder",
    "SatelliteGRUTrainer", "SatelliteAttentionTrainer", "SatelliteFNNTrainer",
    "SatelliteGRU_seq2seq", "SatelliteGRU_attention", "Satellite_FNN",
]