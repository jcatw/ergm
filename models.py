import ergm
import configurations

def p1():
    return ergm.ergm([configurations.n_edges, configurations.n_mutual])

def markov():
    return ergm.ergm([configurations.n_edges,
                      configurations.n_mutual,
                      configurations.two_in_stars,
                      configurations.two_out_stars,
                      configurations.two_mixed_stars,
                      configurations.transitive_triads])

def higher():
    return ergm.ergm([configurations.n_edges,
                      configurations.n_mutual,
                      configurations.geo_in,
                      configurations.geo_out,
                      configurations.transitive_triads,
                      configurations.two_paths,
                      configurations.alternating_k_triangles,
                      configurations.alternating_k_paths])
                      


