import numpy as np

def GetCompCosts(node_dom, configs):
    dom_per_proc = node_dom / configs
    costs = 3.0 * np.multiply.reduce(dom_per_proc, axis=1)
    return costs


def GetAreaNeeded(tgt_area, src_area):
    area_reqd = np.multiply.reduce(tgt_area, axis=1)
    area_intersection = np.multiply.reduce(np.minimum(tgt_area, src_area),
            axis=1)

    return (area_reqd - area_intersection)


def GetCommCosts(src_dom, tgt_dom, src_cfgs, tgt_cfgs):
    m_dim, n_dim, k_dim = 0, 1, 2

    src_dom_per_proc = np.ceil(src_dom, src_cfgs)
    tgt_dom_per_proc = np.ceil(tgt_dom, tgt_cfgs)

    # Cost of communicating input matrix from src to tgt during fwd phase, and
    # from tgt to src during bwd phase
    tgt_area = tgt_dom_per_proc[:, [m_dim,k_dim]]
    src_area = src_dom_per_proc[:, [m_dim,n_dim]]
    area_needed = GetAreaNeeded(tgt_area, src_area)
    costs = 2.0 * np.where(array_needed < 0, 0, array_needed) # Factor 2 is to
                                                             # account for fwd
                                                             # and bwd phases

    # TODO: Add costs when no. of procs in layer1 and layer2 are different


    # Cost for reducing the output during fwd phase
    # All-reduce cost = 2*((m*n)/P)*(P-1)
    words = np.multiply.reduce(tgt_dom_per_proc[:, [m_dim,n_dim]], axis=1)
    procs = tgt_cfgs[:,k_dim]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (words * steps)

    # Cost for gradient update during bwd phase
    # All-reduce cost = 2*((n*k)/P)*(P-1)
    words = np.multiply.reduce(tgt_dom_per_proc[:, [n_dim,k_dim]], axis=1)
    procs = tgt_cfgs[:,m_dim]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (words * steps)

    return costs
