import numpy as np

class BaseMetric:
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    def compute(self, true_graph: np.ndarray, pred_graph: np.ndarray):
        """Computes the metric.

        Args:
            true_graph: The true causal graph as an adjacency matrix.
            pred_graph: The predicted causal graph as an adjacency matrix.

        Returns:
            The computed metric value.
        """
        raise NotImplementedError
