import numpy as np
from .base_metric import BaseMetric

class StructuralHammingDistance(BaseMetric):
    """Computes the Structural Hamming Distance (SHD) of the predicted graph."""

    def __init__(self):
        super().__init__("Structural Hamming Distance")

    def compute(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> int:
        """Computes the Structural Hamming Distance (SHD) of the predicted graph.

        Args:
            true_graph: The true causal graph as an adjacency matrix.
            pred_graph: The predicted causal graph as an adjacency matrix.

        Returns:
            The Structural Hamming Distance of the predicted graph.
        """
        return np.sum(true_graph != pred_graph)
