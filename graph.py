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
pw_ops_in_bn = 9 # No. of pointwise ops in a batch-norm


def factors(n):    
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) +
        1) if n % i == 0)))


def RowCartesian(arr1, arr2):
    shape1 = arr1.shape[0]
    shape2 = arr2.shape[0]

    tile_shape = [shape1] + ([1] * (arr2.ndim - 1))

    arr1 = np.repeat(arr1, repeats=shape2, axis=0)
    arr2 = np.tile(arr2, tile_shape)

    return arr1, arr2


def GetAreaNeeded(src_data_sizes, tgt_data_sizes, src_procs, tgt_procs):
    # Area needed by the target vertex
    tgt_area = np.prod(tgt_data_sizes, axis=1)

    # Intersection of area computed by source, and needed by target.
    # If no. of target procs is more than src procs, then at least one proc
    # contains no source data. So set it to 0.
    area_intersection = np.where(tgt_procs > src_procs, 0,
            np.prod(np.minimum(tgt_data_sizes, src_data_sizes), axis=1))

    # Area that needs to be communicated
    area_needed = tgt_area - area_intersection
    return area_needed


# Returns edge costs for different configs. Edge cost is computed as the
# difference b/w tensor volume needed per proc by the target vertex and the tensor
# volume held per proc by the source vertex.
def GetEdgeCosts(src_op, tgt_op):
    src_tsr = src_op.out_tsr
    tgt_tsr = tgt_op.in_tsr
    src_cfgs = src_op.out_tsr_configs
    tgt_cfgs = tgt_op.in_tsr_configs

    # Calculate the domains per processor
    src_tsr_per_proc = np.ceil(src_tsr / src_cfgs)
    tgt_tsr_per_proc = np.ceil(tgt_tsr / tgt_cfgs)
    src_tsr_per_proc, tgt_tsr_per_proc = RowCartesian(src_tsr_per_proc,
            tgt_tsr_per_proc)

    # Get the no. of procs used for each config
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

    proc_set = []
    for d in node_dom:
        proc_set.append(factors(d))

    configs = [c for c in itertools.product(*proc_set) if reduce(op.mul, c, 1)
        <= n_procs]
    return configs


# Converts convolution domain to GEMM
def ConvToGemm(img, fltr, stride, pad):
    assert(len(img) == 4)
    assert(len(fltr) == 4)
    assert(img[1] == fltr[1])

    c, h, w = img[1], img[2], img[3]
    r, s = fltr[2], fltr[3]

    h_o = int((h - r + 2*pad) / stride) + 1
    w_o = int((w - s + 2*pad) / stride) + 1

    k = c * r * s
    n = fltr[0]
    m = img[0] * h_o * w_o

    return (m, n, k)


class Gemm():
    def __init__(self, dom_size, n_procs, pw_op_cnt=0):
        assert(len(dom_size) == 3)
        self.pw_op_cnt = pw_op_cnt

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

        # Matrix addition cost for weight update
        update_cost = dom_per_proc[:, k_dim] * dom_per_proc[:, n_dim]
        costs += update_cost

        # Cost of pointwise op
        if self.pw_op_cnt > 0:
            pw_cost = dom_per_proc[:, m_dim] * dom_per_proc[:, n_dim]
            costs += self.pw_op_cnt * 3 * pw_cost # Factor 3 is to account for 1
                                                  # pointwise op (per element)
                                                  # in fwd phase, 1
                                                  # differentiation op in bwd
                                                  # phase, and 1 hadamard
                                                  # product in bwd phase
    
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


class MaxPool():
    def __init__(self, fltr, stride):
        assert(len(fltr) == 2)

        self.r = fltr[0]
        self.s = fltr[1]
        self.stride = stride


class Conv():
    def __init__(self, img, fltr, stride, pad, n_procs, maxpool=None, pw_op_cnt=0):
        assert(len(img) == 4)
        assert(len(fltr) == 4)
        assert(img[1] == fltr[1])

        self.n_procs = n_procs
        self.maxpool = maxpool
        self.pw_op_cnt = pw_op_cnt

        m_idx, n_idx, k_idx = 0, 1, 2

        b, n = img[0], fltr[0]
        c, h, w = img[1], img[2], img[3]
        r, s = fltr[2], fltr[3]

        h_o = int((h - r + 2*pad) / stride) + 1
        w_o = int((w - s + 2*pad) / stride) + 1

        # Domain
        self.dom = ConvToGemm(img, fltr, stride, pad)

        # Reduced domain when maxpool is applied to 'img'. Configurations are
        # created on this reduced domain so that there is no intra-op
        # communication for max-pooling.
        if maxpool:
            self.maxpool.h_o = int((h_o - maxpool.r) / maxpool.stride) + 1
            self.maxpool.w_o = int((w_o - maxpool.s) / maxpool.stride) + 1

            maxpool_m = b * self.maxpool.h_o * self.maxpool.w_o
            self.maxpool.dom = (maxpool_m, n)
            dom_with_maxpool = [maxpool_m] + self.dom[1:]
        else:
            dom_with_maxpool = self.dom

        # Input/output tensors
        self.in_tsr = (self.dom[m_idx], self.dom[k_idx])
        if maxpool:
            self.out_tsr = self.maxpool.dom
            self.orig_out_tsr = (b, n, self.maxpool.h_o, self.maxpool.w_o)
        else:
            self.out_tsr = (self.dom[m_idx], self.dom[n_idx])
            self.orig_out_tsr = (b, n, h_o, w_o)

        # Configurations
        self.dom_config_tuples = GetNodeConfigs(dom_with_maxpool, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs[:, (m_idx, k_idx)]
        self.out_tsr_configs = self.dom_configs[:, m_idx:n_idx+1]
        if maxpool:
            self.maxpool.dom_configs = self.dom_configs[:, 0:2]


    def GetVertexCosts(self):
        gemm_op = Gemm(self.dom, self.n_procs, self.pw_op_cnt)
        costs = gemm_op.GetVertexCosts()

        # Add the cost for max-pooling
        if self.maxpool:
            dom_per_proc = self.maxpool.dom / self.maxpool.dom_configs
            maxpool_costs = 3.0 * np.prod(dom_per_proc, axis=1)

            assert(costs.shape == maxpool_costs.shape)
            costs += maxpool_costs

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


def Alexnet(G, b, n_procs):
    img = (b, 3, 227, 227)
    node_id = 0

    # Conv1 + relu + maxpool + norm
    dom = ConvToGemm(img, (96, 3, 11, 11), 4, 0)
    node_op = Conv(img, (96, 3, 11, 11), 4, 0, n_procs, MaxPool((3,3), 2),
            pw_ops_in_bn+1)
    G.add_node(node_id, op=node_op)
    node_id += 1

    # Conv2 + relu + maxpool + norm
    dom = ConvToGemm(dom, (256, 96, 5, 5), 1, 2)
    node_op = Conv(node_op.orig_out_tsr, (256, 96, 5, 5), 1, 2, n_procs,
            MaxPool((3,3), 2), pw_ops_in_bn+1)
    G.add_node(node_id, op=node_op)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # Conv3 + relu
    dom = ConvToGemm(dom, (384, 256, 3, 3), 1, 1)
    node_op = Conv(node_op.orig_out_tsr, (384, 256, 3, 3), 1, 1, n_procs,
            pw_op_cnt=1)
    G.add_node(node_id, op=node_op)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # Conv4 + relu
    dom = ConvToGemm(dom, (384, 384, 3, 3), 1, 1)
    node_op = Conv(node_op.orig_out_tsr, (384, 384, 3, 3), 1, 1, n_procs,
            pw_op_cnt=1)
    G.add_node(node_id, op=node_op)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # Conv5 + relu + maxpool
    dom = ConvToGemm(dom, (256, 384, 3, 3), 1, 1)
    node_op = Conv(node_op.orig_out_tsr, (256, 384, 3, 3), 1, 1, n_procs,
            MaxPool((3,3), 2), 1)
    G.add_node(node_id, op=node_op)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # TODO: Reshape

    # FC6 + relu
    node_op = Gemm(node_op.out_tsr, (node_op.out_tsr[-1], 4096), n_procs, 1)
    G.add_node(node_id, op=node_op)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # FC7 + relu
    node_op = Gemm(node_op.out_tsr, (node_op.out_tsr[-1], 4096), n_procs, 1)
    G.add_node(node_id, op=node_op)
    G.add_edge(node_id - 1, node_id)
    node_id += 1

    # FC8 + relu
    node_op = Gemm(node_op.out_tsr, (node_op.out_tsr[-1], 1024), n_procs, 1)
    G.add_node(node_id, op=node_op)
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


