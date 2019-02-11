import numpy as np


def RowCartesian(arr1, arr2):
    shape1 = arr1.shape[0]
    shape2 = arr2.shape[0]

    arr1 = np.repeat(arr1, repeats=shape2, axis=0)
    arr2 = np.tile(arr2, (shape1, 1))

    return arr1, arr2


# Returns vertex costs for different configs
def GetVertexCosts(node_dom, configs):
    m_dim, n_dim, k_dim = 0, 1, 2

    dom_per_proc = np.ceil(node_dom / configs)
    costs = 3.0 * np.multiply.reduce(dom_per_proc, axis=1)

    # Cost for reducing the output during fwd phase
    # All-reduce cost = 2*((m*n)/P)*(P-1)
    words = np.multiply.reduce(dom_per_proc[:, m_dim:n_dim+1], axis=1)
    procs = configs[:,k_dim]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (words * steps)

    # Cost for gradient update during bwd phase
    # All-reduce cost = 2*((n*k)/P)*(P-1)
    words = np.multiply.reduce(dom_per_proc[:, [n_dim,k_dim]], axis=1)
    procs = configs[:,m_dim]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (words * steps)

    return costs


def GetAreaNeeded(tgt_area, src_area):
    area_reqd = np.multiply.reduce(tgt_area, axis=1)
    area_intersection = np.multiply.reduce(np.minimum(tgt_area, src_area),
            axis=1)

    return (area_reqd - area_intersection)


# Returns edge costs for different configs
def GetEdgeCosts(src_dom, tgt_dom, src_cfgs, tgt_cfgs):
    m_dim, n_dim, k_dim = 0, 1, 2

    src_dom_per_proc = np.ceil(src_dom / src_cfgs)
    tgt_dom_per_proc = np.ceil(tgt_dom / tgt_cfgs)

    # Cost of communicating input matrix from src to tgt during fwd phase, and
    # from tgt to src during bwd phase
    src_dom_per_proc, tgt_dom_per_proc = RowCartesian(src_dom_per_proc[:,
        m_dim:n_dim+1], tgt_dom_per_proc[:, [m_dim,k_dim]])
    area_needed = GetAreaNeeded(tgt_dom_per_proc, src_dom_per_proc)
    costs = 2.0 * np.where(area_needed < 0, 0, area_needed) # Factor 2 is to
                                                            # account for fwd
                                                            # and bwd phases

    # TODO: Add costs when no. of procs in layer1 and layer2 are different

    return costs
