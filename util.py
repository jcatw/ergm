"""
Utility functions for the ergm package.
"""

import numpy as np

def pam(fs, x):
    """
    Apply a sequence of functions to x.  

    Args:
      fs: a sequence of functions.
      x: a value to which each f in fs can be applied.

    Returns:
      A list of the result of applying each f to x.
    """
    return [f(x) for f in fs]

def permute(G):
    """
    Invert an edge in G uniformly at random and return the
    result.  Does not mutate G.

    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A 2d numpy array representing an adjacency matrix of the permuted graph.
    """
    permuter = np.zeros(G.shape)

    n_nodes = G.shape[0]
    node_from = np.random.randint(n_nodes)
    node_to = np.random.randint(n_nodes)
    permuter[node_from, node_to] = 1

    return np.logical_xor(G, permuter).astype(G.dtype)
        
