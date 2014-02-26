"""
A pure-python exponential random graph model (ERGM) implementation.

classes:
    ergm
"""
import numpy as np
import util


class ergm:
    def __init__(self, features):
        """
        Construct an ergm over a set of features.

        Args: 
          features: a list of functions which compute the *count*
                    of some feature.  These functions should take a 2d
                    numpy array representing an adjacnecy matrix as input and
                    return a numeric value.
        """
        
        self.features = features
        self.coefs = [0.] * len(features)

    def weight(self, G):
        """
        Compute the weight of a graph.  The weight is proportional to the
        probability of the graph.

        Args:
          G: A 2d numpy array representing an adjacency matrix.

        Returns:
          A float.
        """
        
        return np.exp( np.sum( util.pam(self.features, G) ))

    def sum_weights(self, X):
        """
        Compute the total weight associated with graphs X.  Each row
        of X represents one graph.

        Args:
          X: A 2d numpy array representing a set of graphs.

        Returns:
          A float.
        """
        n_nodes = int(np.sqrt(X.shape[1]))
        return np.sum([self.weight(X[i].reshape((n_nodes, n_nodes))) for i in xrange(X.shape[0])])

    def fit(self, X, n_iterations=100, n_graph_samples=1000, alpha=0.01):
        """
        Fit the ergm via Metropolis-Hastings MCMC.  This alters self.coefs.

        Args:
          X: A 2d numpy array representing a set of graphs.
          n_iterations: The number of MCMC iterations to take.
          n_graph_samples: The number of samples to draw when fitting.
          alpha: How much should we jump around the parameter space?

        Returns:
          A list of floats representing the most likely coefficients.
        """
        n_input_graphs = X.shape[0]
        n_nodes = int(np.sqrt(X.shape[1]))
        ps = [None]
        all_coefs = [self.coefs]
        
        for i in xrange(n_iterations):
            self.coefs = [np.random.normal(x,alpha) for x in all_coefs[-1]]
            # sample graphs of the same size as the input
            graph_samples = self.sample(n_nodes, n_graph_samples)
            kappa = self.sum_weights(graph_samples)
            p = np.product([self.weight(X[j].reshape((n_nodes,n_nodes))) / kappa
                            for j in xrange(n_input_graphs)])
            u = np.random.rand()
            if p > ps[-1] or u < (p / ps[-1]):
                all_coefs.append(self.coefs)
                ps.append(p)
            else:
                all_coefs.append(all_coefs[-1])
                ps.append(None)

        # set this ergm's coefficients to the most likely values
        index = np.argmax(ps)
        self.coefs = all_coefs[index]

        return self.coefs
        

    def sample(self, n_nodes, n_samples):
        """
        Sample n graphs from this ergm via Metropolis-Hastings MCMC.

        Args:
          n: The number of samples to generate.

        Returns:
          X: A n_samples x n_nodes^2 2d numpy array representing the sampled graphs.  Each
             row in X represents a single graph.
        """
        
        samples = np.zeros((n_samples, n_nodes**2))
        # start from a random adjacency matrix
        this_G = np.random.randint(0, 1, (n_nodes, n_nodes))
        
        this_w = self.weight(this_G)

        i = 0
        while i < n_samples:
            new_G = util.permute(this_G)
            new_w = self.weight(new_G)
            u = np.random.rand()
            if new_w > this_w or u < (new_w/this_w):
                samples[i] = new_G.reshape(n_nodes**2)
                this_G = new_G
                this_w = new_w
                i += 1

        return samples
        
