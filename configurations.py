"""
Useful configuration functions for defining ERGMs.
"""

import numpy as np

def n_edges(G):
    """
    Compute the number of edges in G.
    
    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A float.
    """
    return np.sum(G)

def n_mutual(G):
    """
    Compute the number of mutual edges in G.
    
    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A float.
    """
    sum = 0.
    for i in xrange(G.shape[0]):
        for j in xrange(G.shape[1]):
            sum += G[i,j] * G[j,i]
    return sum
    
