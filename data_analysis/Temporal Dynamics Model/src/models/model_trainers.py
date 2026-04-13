"""
Model Trainers - Backward Compatibility Module

This module provides backward compatibility by importing all trainers
from the organized models directory.
"""

from .var_model import VARTrainer
from .lstm_model import LSTMTrainer, SimpleLSTM
from .transformer_model import TransformerTrainer, SimpleTransformer
from .wavenet_model import WaveNetTrainer, WaveNet
from .gaussian_process_model import GaussianProcessTrainer
from .autoencoder_models import (
    PredictiveAutoencoderTrainer,
    PredictiveAutoencoder,
    SparseAutoencoder,
    VariationalAutoencoder,
)
from .varima_model import VARIMATrainer

__all__ = [
    "VARTrainer",
    "LSTMTrainer",
    "TransformerTrainer",
    "WaveNetTrainer",
    "GaussianProcessTrainer",
    "PredictiveAutoencoderTrainer",
    "VARIMATrainer",
    "SimpleLSTM",
    "SimpleTransformer",
    "WaveNet",
    "PredictiveAutoencoder",
    "SparseAutoencoder",
    "VariationalAutoencoder",
]
