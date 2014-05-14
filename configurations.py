"""
Useful configuration functions for defining ERGMs.
"""

import numpy as np
from scipy.misc import comb
import itertools

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
    
def two_in_stars(G):
    """
    Compute the number of two-in-stars in G.

    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A float.
    """
    sum = 0.
    incomings = G.sum(0) #sum over rows
    for incoming in incomings:
        sum += comb(incoming, 2)
    return sum

def two_out_stars(G):
    """
    Compute the number of two-out-stars in G.

    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A float.
    """
    sum = 0.
    outgoings = G.sum(1) #sum over cols
    for outgoing in outgoings:
        sum += comb(outgoing, 2)
    return sum

def two_mixed_stars(G):
    """
    Compute the number of two-mixed-stars in G.

    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A float.
    """
    sum = 0.
    incomings = G.sum(0)
    outgoings = G.sum(1)
    for incoming, outgoing in zip(incomings, outgoings):
        sum += incoming * outgoing
    return sum

def cyclic_triads(G):
    """
    Compute the number of cyclic triads in G.

    Args:
      G: A 2d numpy array representing an adjacency matrix.

    Returns:
      A float.
    """
    return ((G.dot(G)).dot(G)).sum() / 3.

def transitive_triads(G):
    n_nodes = G.shape[0]
    clust_coef = np.zeros(n_nodes)

    for i in xrange(n_nodes):
        neighbors = np.where(G[:,i]==1)[0]
        n_neighbors = neighbors.shape[0]
        if n_neighbors == 0:
            clust_coef[i] = 0
        else:
            transitive_neighbors = 0.
            for (edge_from, edge_to) in itertools.combinations(neighbors,2):
                if G[edge_from, edge_to] == 1 or G[edge_to,edge_from] == 1:
                    transitive_neighbors += 1
            clust_coef[i] = transitive_neighbors
    return clust_coef.sum()

def geo_out(G):
    out_degrees = G.sum(1) #sum over cols
    max_out = max(out_degrees)
    out_dist = np.zeros(int(max_out) + 1)

    for d in out_degrees:
        out_dist[int(d)] += 1.
        
    geo_out_dist = np.sum([np.exp(-i) * d for i,d in enumerate(out_dist)])

    return geo_out_dist

def geo_in(G):
    in_degrees = G.sum(0) #sum over rows
    max_in = max(in_degrees)
    in_dist = np.zeros(int(max_in) + 1)

    for d in in_degrees:
        in_dist[int(d)] += 1.
        
    geo_in_dist = np.sum([np.exp(-i) * d for i,d in enumerate(in_dist)])

    return geo_in_dist

def transitive_triads(G):
    sum = 0.
    n_nodes = G.shape[0]
    for i in xrange(n_nodes):
        for j in xrange(n_nodes):
            for k in xrange(n_nodes):
                sum += G[i,j] * G[j,k] * G[i,k]
    return sum
    
def two_paths(G):
    sum = 0.
    n_nodes = G.shape[0]
    for i in xrange(n_nodes):
        for j in xrange(n_nodes):
            for k in xrange(n_nodes):
                if i != k and j != k:
                    sum += G[i,k] * G[k,j]
    return sum

def alternating_k_triangles(G):
    sum = 0.
    n_nodes = G.shape[0]
    for i in xrange(n_nodes):
        for j in xrange(n_nodes):
            two_paths = 0
            for k in xrange(n_nodes):
                if i != k and j != k:
                    two_paths += G[i,k] * G[k,j]
            sum += 2 * G[i,j] * (1 - (0.5 ** two_paths))
    return sum

def alternating_k_paths(G):
    sum = 0.
    n_nodes = G.shape[0]
    for i in xrange(n_nodes):
        for j in xrange(n_nodes):
            two_paths = 0
            for k in xrange(n_nodes):
                if i != k and j != k:
                    two_paths += G[i,k] * G[k,j]
            sum += 2 * (1 - (0.5 ** two_paths))
    return sum
