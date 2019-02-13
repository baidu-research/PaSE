import numpy as np


peak_flop = float(15 * 1000) # 125 TFLOP
bw = float(150 / 8) # 150 GBytes/sec = 7/8 GWords/sec
bw_to_flop = float(peak_flop / bw)


def RowCartesian(arr1, arr2):
    shape1 = arr1.shape[0]
    shape2 = arr2.shape[0]

    tile_shape = [shape1] + ([1] * (arr2.ndim - 1))

    arr1 = np.repeat(arr1, repeats=shape2, axis=0)
    arr2 = np.tile(arr2, tile_shape)

    return arr1, arr2


# Returns vertex costs for different configs
def GetVertexCosts(node_dom, configs):
    m_dim, n_dim, k_dim = 0, 1, 2

    dom_per_proc = np.ceil(node_dom / configs)
    costs = 3.0 * np.prod(dom_per_proc, axis=1)

    # Cost for reducing the output during fwd phase
    # All-reduce cost = 2*((m*n)/P)*(P-1)
    words = np.prod(dom_per_proc[:, m_dim:n_dim+1], axis=1)
    procs = configs[:,k_dim]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (bw_to_flop * (words * steps))

    # Cost for gradient update during bwd phase
    # All-reduce cost = 2*((n*k)/P)*(P-1)
    words = np.prod(dom_per_proc[:, [n_dim,k_dim]], axis=1)
    procs = configs[:,m_dim]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (bw_to_flop * (words * steps))

    return costs


def GetAreaNeeded(src_data_sizes, tgt_data_sizes, src_procs, tgt_procs):
    # Area needed by the target vertex
    area_reqd = np.prod(tgt_data_sizes, axis=1)

    # Intersection of area computed by source, and needed by target.
    # If no. of target procs is more than src procs, then at least one proc
    # contains no source data. So set it to 0
    area_intersection = np.where(tgt_procs > src_procs, 0,
            np.prod(np.minimum(tgt_data_sizes, src_data_sizes), axis=1))

    # Area that needs to be communicated
    area_needed = area_reqd - area_intersection
    return area_needed


# Returns edge costs for different configs
def GetEdgeCosts(src_dom, tgt_dom, src_cfgs, tgt_cfgs):
    m_dim, n_dim, k_dim = 0, 1, 2

    src_dom_per_proc = np.ceil(src_dom / src_cfgs)
    tgt_dom_per_proc = np.ceil(tgt_dom / tgt_cfgs)
    src_dom_per_proc, tgt_dom_per_proc = RowCartesian(src_dom_per_proc[:,
        m_dim:n_dim+1], tgt_dom_per_proc[:, [m_dim,k_dim]])

    src_procs = np.prod(src_cfgs, axis=1)
    tgt_procs = np.prod(tgt_cfgs, axis=1)
    src_procs, tgt_procs = RowCartesian(src_procs, tgt_procs)

    # Cost of communicating input matrix from src to tgt during fwd phase, and
    # from tgt to src during bwd phase
    area_needed = GetAreaNeeded(src_dom_per_proc, tgt_dom_per_proc, src_procs,
            tgt_procs)
    costs = 2.0 * np.where(area_needed < 0, 0, area_needed) # Factor 2 is to
                                                            # account for fwd
                                                            # and bwd phases
    costs *= bw_to_flop

    return costs
