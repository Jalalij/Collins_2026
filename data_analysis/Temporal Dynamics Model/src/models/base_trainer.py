# base_trainer.py
from typing import Dict, Any, Tuple


class BaseTrainer:
    """
    Abstract base trainer class.
    All trainers must implement train_and_evaluate with this signature.
    """

    def train_and_evaluate(
        self,
        train_dataset,  # ExperimentDataset
        test_dataset,  # ExperimentDataset
        evaluator,
        results_dir: str,
        direction: str = "UL",
    ) -> Tuple[Dict[str, Any], Dict, Dict]:
        raise NotImplementedError("train_and_evaluate must be implemented in subclass")
