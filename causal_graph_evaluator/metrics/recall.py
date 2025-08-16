import numpy as np
from .base_metric import BaseMetric

class Recall(BaseMetric):
    """Computes the recall of the predicted graph."""

    def __init__(self):
        super().__init__("Recall")

    def compute(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> float:
        """Computes the recall of the predicted graph.

        Args:
            true_graph: The true causal graph as an adjacency matrix.
            pred_graph: The predicted causal graph as an adjacency matrix.

        Returns:
            The recall of the predicted graph.
        """
        true_positives = np.sum((pred_graph == 1) & (true_graph == 1))
        actual_positives = np.sum(true_graph == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0
