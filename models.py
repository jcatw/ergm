import ergm
import configurations

def p1():
    return ergm.ergm([configurations.n_edges, configurations.n_mutual])

