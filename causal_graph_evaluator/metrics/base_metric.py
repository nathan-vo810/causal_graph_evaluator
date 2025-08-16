from typing import List, Tuple, Union
import numpy as np
from .utils import to_adjacency_matrix

class BaseMetric:
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, 
                 true_graph: Union[np.ndarray, List[Tuple[int, int]]], 
                 pred_graph: Union[np.ndarray, List[Tuple[int, int]]], 
                 num_nodes: int = None):
        """
        Calculates the metric score between two graphs.
        The graphs can be either adjacency matrices or edge lists.
        """
        true_adj = to_adjacency_matrix(true_graph, num_nodes)
        pred_adj = to_adjacency_matrix(pred_graph, num_nodes)
        
        if true_adj.shape != pred_adj.shape:
            raise ValueError("The shape of the true and predicted graphs must be the same.")

        return self.compute(true_adj, pred_adj)

    def compute(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> float:
        """
        Computes the metric.
        This method should be implemented by the subclasses.
        """
        raise NotImplementedError
