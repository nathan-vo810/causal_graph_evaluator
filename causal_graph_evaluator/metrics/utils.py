import numpy as np
from typing import List, Tuple, Union

def to_adjacency_matrix(graph: Union[np.ndarray, List[Tuple[int, int]]], num_nodes: int = None) -> np.ndarray:
    """Converts a graph from edge list format to an adjacency matrix.

    If the graph is already an adjacency matrix, it is returned unchanged.

    Args:
        graph: The graph to convert, either as an adjacency matrix or an edge list.
        num_nodes: The number of nodes in the graph. If not provided, it's inferred
                   from the maximum node index in the edge list.

    Returns:
        The graph as an adjacency matrix.
    """
    if isinstance(graph, np.ndarray):
        return graph

    if num_nodes is None:
        if not graph:
            num_nodes = 0
        else:
            num_nodes = max(max(edge) for edge in graph) + 1

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i, j in graph:
        adj_matrix[i, j] = 1
    return adj_matrix
