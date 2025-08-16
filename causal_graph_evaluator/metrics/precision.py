import numpy as np
from .base_metric import BaseMetric

class Precision(BaseMetric):
    """Computes the precision of the predicted graph."""

    def __init__(self):
        super().__init__("Precision")

    def compute(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> float:
        """Computes the precision of the predicted graph.

        Args:
            true_graph: The true causal graph as an adjacency matrix.
            pred_graph: The predicted causal graph as an adjacency matrix.

        Returns:
            The precision of the predicted graph.
        """
        true_positives = np.sum((pred_graph == 1) & (true_graph == 1))
        predicted_positives = np.sum(pred_graph == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
