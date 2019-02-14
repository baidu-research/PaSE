import networkx as nx
import numpy as np
import pandas as pd

import math
import itertools
from functools import reduce
import operator as op


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
def GetEdgeCosts(src_op, tgt_op):
    src_tsr = src_op.out_tsr
    tgt_tsr = tgt_op.in_tsr
    src_cfgs = src_op.out_tsr_configs
    tgt_cfgs = tgt_op.in_tsr_configs

    src_tsr_per_proc = np.ceil(src_tsr / src_cfgs)
    tgt_tsr_per_proc = np.ceil(tgt_tsr / tgt_cfgs)
    src_tsr_per_proc, tgt_tsr_per_proc = RowCartesian(src_tsr_per_proc,
            tgt_tsr_per_proc)

    src_procs = np.prod(src_cfgs, axis=1)
    tgt_procs = np.prod(tgt_cfgs, axis=1)
    src_procs, tgt_procs = RowCartesian(src_procs, tgt_procs)

    # Cost of communicating input matrix from src to tgt during fwd phase, and
    # from tgt to src during bwd phase
    area_needed = GetAreaNeeded(src_tsr_per_proc, tgt_tsr_per_proc, src_procs,
            tgt_procs)
    costs = 2.0 * np.where(area_needed < 0, 0, area_needed) # Factor 2 is to
                                                            # account for fwd
                                                            # and bwd phases
    costs *= bw_to_flop

    return costs


# Generates list of configurations for a vertex
def GetNodeConfigs(node_dom, n_procs):
    dim = len(node_dom)
    log_n_procs = int(math.log2(n_procs))
    procs = [1 << i for i in range(log_n_procs + 1)]

    configs = [c for c in itertools.product(procs, repeat=dim) if reduce(op.mul,
        c, 1) <= n_procs]
    return configs


class Gemm():
    def __init__(self, dom_size, n_procs, pw_op=False):
        assert(len(dom_size) == 3)
        self.pw_op = pw_op

        m_idx, n_idx, k_idx = 0, 1, 2

        # Domain and input/output tensors
        self.dom = tuple(dom_size)
        self.in_tsr = (dom_size[m_idx], dom_size[k_idx])
        self.out_tsr = (dom_size[m_idx], dom_size[n_idx])

        # Configurations
        self.dom_config_tuples = GetNodeConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs[:, (m_idx, k_idx)]
        self.out_tsr_configs = self.dom_configs[:, m_idx:n_idx+1]


    # Returns vertex costs for different configs
    def GetVertexCosts(self):
        m_dim, n_dim, k_dim = 0, 1, 2
 
        # Cost for 1 GEMM in fwd phase + 2 GEMMs in bwd phase
        dom_per_proc = np.ceil(self.dom / self.dom_configs)
        costs = 3.0 * np.prod(dom_per_proc, axis=1)

        # Cost of pointwise op
        if self.pw_op == True:
            costs += (dom_per_proc[:, m_dim] * dom_per_proc[:, n_dim])
    
        # Cost for reducing the output during fwd phase
        # All-reduce cost = 2*((m*n)/P)*(P-1)
        words = np.prod(dom_per_proc[:, m_dim:n_dim+1], axis=1)
        procs = self.dom_configs[:,k_dim]
        words /= procs
        steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
        costs += (bw_to_flop * (words * steps))
    
        # Cost for gradient update during bwd phase
        # All-reduce cost = 2*((n*k)/P)*(P-1)
        words = np.prod(dom_per_proc[:, [n_dim,k_dim]], axis=1)
        procs = self.dom_configs[:,m_dim]
        words /= procs
        steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
        costs += (bw_to_flop * (words * steps))
    
        return costs


# Computes vertex costs for different configurations of vertices, and assigns
# them to the vertices
def AssignCostsToNodes(G):
    for v, attr in G.nodes(data=True):
        op = attr['op']
        costs = op.GetVertexCosts()
        idx = pd.Index(op.dom_configs, dtype=tuple, name=str(v))
        attr['costs'] = pd.Series(costs, index=idx, name='cost')


# Computes edge costs for different configs, and assigns them to edges
def AssignCostsToEdges(G):
    node_ops = G.nodes(data='op')

    for src, tgt, edge_attr in G.edges(data=True):
        src_op = node_ops[src]
        tgt_op = node_ops[tgt]
        src_config_tuples = src_op.dom_config_tuples
        tgt_config_tuples = tgt_op.dom_config_tuples

        costs = GetEdgeCosts(src_op, tgt_op)
        idx = pd.MultiIndex.from_product([src_config_tuples, tgt_config_tuples],
                names=[str(src), str(tgt)])
        edge_attr['costs'] = pd.Series(costs, index=idx, name='cost')


# Converts convolution domain to GEMM
def ConvToGemm(img, fltr, stride, pad):
    assert(len(img) == 4)
    assert(len(fltr) == 4)
    assert(img[1] == fltr[1])

    c = img[1]
    h = img[2]
    w = img[3]
    r = fltr[2]
    s = fltr[3]

    h_o = int((h - r + 2*pad) / stride) + 1
    w_o = int((w - s + 2*pad) / stride) + 1

    k = c * r * s
    n = fltr[0]
    m = img[0] * h_o * w_o

    return [m, n, k]


def Alexnet(G, b, n_procs):
    img = [b, 3, 227, 227]
    node_id = 0

    # Conv1
    dom = ConvToGemm(img, [96, 3, 11, 11], 4, 0)
    G.add_node(node_id, dim=len(dom), dom=dom)
    node_id += 1

    # TODO: maxpool

    # Conv2
    dom = ConvToGemm(dom, [256, 96, 5, 5], 1, 2)
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # TODO: maxpool

    # Conv3
    dom = ConvToGemm(dom, [384, 256, 3, 3], 1, 1)
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # Conv4
    dom = ConvToGemm(dom, [384, 384, 3, 3], 1, 1)
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # Conv5
    dom = ConvToGemm(dom, [256, 384, 3, 3], 1, 1)
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # TODO: maxpool

    # FC1
    dom = [b, 4096, 256*6*6]
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # FC2
    dom = [b, 4096, 4096]
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # FC3
    dom = [b, 1024, 4096]
    G.add_node(node_id, dim=len(dom), dom=dom)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    return G


def TestGraph(G, batch_size, hidden_dim_size, n_procs):
    dom = [batch_size, hidden_dim_size, hidden_dim_size]

    G.add_node(0, op=Gemm(dom, n_procs))
    G.add_node(1, op=Gemm(dom, n_procs))
    G.add_node(2, op=Gemm(dom, n_procs))

    G.add_edge(0, 1)
    G.add_edge(1, 2)

    return G


# Creates the graph for the model
def CreateGraph(graph_type, batch_size, hidden_dim_size, n_procs):
    G = nx.DiGraph()

    if graph_type == 'test':
        G = TestGraph(G, batch_size, hidden_dim_size, n_procs)
    elif graph_type == 'alexnet':
        G = Alexnet(G, batch_size, n_procs)
    else:
        assert(False)

    return G


