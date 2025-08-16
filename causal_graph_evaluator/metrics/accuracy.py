import numpy as np
from .base_metric import BaseMetric

class Accuracy(BaseMetric):
    """Computes the accuracy of the predicted graph."""

    def __init__(self):
        super().__init__("Accuracy")

    def compute(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> float:
        """Computes the accuracy of the predicted graph.

        Args:
            true_graph: The true causal graph as an adjacency matrix.
            pred_graph: The predicted causal graph as an adjacency matrix.

        Returns:
            The accuracy of the predicted graph.
        """
        return np.sum(true_graph == pred_graph) / true_graph.size
