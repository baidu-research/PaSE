import networkx as nx
import numpy as np
import pandas as pd

import math
import itertools
from functools import reduce
import operator as op


p100_peak_flop = float(10.6 * 1000) # GFLOPs
p100_bw = float((36.72 * 2) / 8) # Unidirectional for 2 sublinks per direction.
                                 # GBytes/sec = b/8 GWords/sec

v100_peak_flop = float(15.7 * 1000) # GFLOPs
v100_bw = float((47.99 * 3) / 8) # Unidirectional for 3 sublinks per direction.
                                 # GBytes/sec = b/8 GWords/sec

peak_flop = p100_peak_flop
bw = p100_bw
bw_to_flop = float(peak_flop / bw)
pw_ops_in_bn = 9 # No. of pointwise ops in a batch-norm
exp_cost = 3 # Cost of computing exponent


def MakePair(v):
    if hasattr(v, "__len__"):
        assert(len(v) == 2)
        return v
    else:
        return (v, v)


def factors(n):
    assert(n > 0)
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) +
        1) if n % i == 0)))


def RowCartesianProd(arr1, arr2):
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


def Reshape(src_tsr, tgt_tsr, src_tsr_per_proc, tgt_tsr_per_proc, reshape):
    assert(len(src_tsr) == reduce(op.add, [len(i) for i in reshape]))
    assert(len(tgt_tsr) == len(reshape))

    sz = tgt_tsr_per_proc.shape[0]
    reshaped_tsr_per_proc = np.empty([sz, src_tsr_per_proc.shape[1]])
    for i, s in enumerate(reshape):
        if len(s) == 1:
            reshaped_tsr_per_proc[:, s[0]] = tgt_tsr_per_proc[:, i]
        else:
            arr = np.empty([sz, len(s)])
            arr[:,-1] = tgt_tsr_per_proc[:, i]

            for j in range(len(s)-1, 0, -1):
                idx = s[j]
                val = src_tsr[idx]

                assert(idx < len(src_tsr))
                #assert(np.logical_or((arr[:, j] % val == 0), (val % arr[:,
                #    j] == 0)).all())

                #arr[:, j-1] = np.maximum(1, arr[:, j] // val)
                arr[:, j-1] = np.ceil(arr[:,j] / val).astype(int)
                arr[:, j] = np.where(arr[:,j] > val, val, arr[:,j])

            reshaped_tsr_per_proc[:, s] = arr

    return reshaped_tsr_per_proc


def GetConvolutedSize(h, w, r, s, stride, pad):
    stride_r, stride_s = MakePair(stride)
    pad_r, pad_s = MakePair(pad)

    h_o = int((h - r + 2*pad_r) / stride_r) + 1
    w_o = int((w - s + 2*pad_s) / stride_s) + 1

    return h_o, w_o


# Make sure size of all dimensions except 'concat_dim' is the same for all
# in_ops
def CheckConcat(in_ops, concat_dim):
    l = len(in_ops[0].out_tsr)
    for i, t in enumerate(in_ops):
        assert(concat_dim < len(t.out_tsr))
        assert(len(t.out_tsr) == l)
        for j in range(l):
            if j != concat_dim:
                assert(t.out_tsr[j] == in_ops[0].out_tsr[j])


def GetConcatenatedSize(in_ops, concat_dim):
    CheckConcat(in_ops, concat_dim)
    assert(len(in_ops) > 1)

    out_tsr = list(in_ops[0].out_tsr)
    for i in range(1, len(in_ops)):
        out_tsr[concat_dim] += in_ops[i].out_tsr[concat_dim]

    return out_tsr


# Returns edge costs for different configs. Edge cost is computed as the
# difference b/w tensor volume needed per proc by the target vertex and the tensor
# volume held per proc by the source vertex.
def GetEdgeCosts(src_tsr, tgt_tsr, src_cfgs, tgt_cfgs, reshape):
    # Calculate the domains per processor
    src_tsr_per_proc = np.ceil(src_tsr / src_cfgs)
    tgt_tsr_per_proc = np.ceil(tgt_tsr / tgt_cfgs)

    if reshape:
        tgt_tsr_per_proc = Reshape(src_tsr, tgt_tsr, src_tsr_per_proc,
                tgt_tsr_per_proc, reshape)
    else:
        assert(len(src_tsr) == len(tgt_tsr))

    src_tsr_per_proc, tgt_tsr_per_proc = RowCartesianProd(src_tsr_per_proc,
            tgt_tsr_per_proc)

    # Get the no. of procs used for each config
    src_procs = np.prod(src_cfgs, axis=1)
    tgt_procs = np.prod(tgt_cfgs, axis=1)
    src_procs, tgt_procs = RowCartesianProd(src_procs, tgt_procs)

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
def GetConfigs(vol, n_procs):
    cutoff = 4 # Minimum domain size to reduce search space
    dim = len(vol)

    proc_set = []
    for d in vol:
        s = factors(d)
        l = [e for e in s if d/e >= cutoff]
        if len(l) <= 0:
            l = [1]
        proc_set.append(l)

    configs = [c for c in itertools.product(*proc_set) if reduce(op.mul, c, 1)
        <= n_procs]
    return configs


def ComputeGemmCost(dom, dom_configs, pw_op_cnt):
    m_idx, n_idx, k_idx = 0, 1, 2
    dom_per_proc = np.ceil(dom / dom_configs)

    # Cost for 1 GEMM in fwd phase + 2 GEMMs in bwd phase
    costs = 3.0 * np.prod(dom_per_proc, axis=1)

    # Matrix addition cost for weight update
    update_cost = dom_per_proc[:, k_idx] * dom_per_proc[:, n_idx]
    costs += update_cost

    # Cost of pointwise op
    if pw_op_cnt > 0:
        pw_cost = dom_per_proc[:, m_idx] * dom_per_proc[:, n_idx]
        costs += pw_op_cnt * 3 * pw_cost # Factor 3 is to
                                         # account for 1 pointwise op (per
                                         # element) in fwd phase, 1
                                         # differentiation op in bwd phase, and
                                         # 1 hadamard product in bwd phase
    
    # Cost for reducing the output during fwd phase
    # All-reduce cost = 2*((m*n)/P)*(P-1)
    words = np.prod(dom_per_proc[:, m_idx:n_idx+1], axis=1)
    procs = dom_configs[:,k_idx]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (bw_to_flop * (words * steps))
    
    # Cost for gradient update during bwd phase
    # All-reduce cost = 2*((n*k)/P)*(P-1)
    words = np.prod(dom_per_proc[:, [n_idx,k_idx]], axis=1)
    procs = dom_configs[:,m_idx]
    words /= procs
    steps = 2.0 * (procs - 1) # When procs = 1, the cost is 0
    costs += (bw_to_flop * (words * steps))

    return costs


class Embedding():
    def __init__(self, b, vocab_size, embed_dim, n_procs):
        self.dom = (b,)
        self.in_tsr = (b, vocab_size)
        self.out_tsr = (b, embed_dim)

        self.dom_config_tuples = GetConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples).reshape(-1,1)
        rep = np.repeat(1, self.dom_configs.shape[0]).reshape(-1, 1)
        self.in_tsr_configs = np.concatenate((self.dom_configs, rep), axis=1)
        self.out_tsr_configs = self.in_tsr_configs

        # Backprop cost of synchronization
        dom_per_proc = np.ceil(self.dom / self.dom_configs)
        words = np.prod(dom_per_proc, axis = 1)
        procs = self.dom_configs[:,0]
        words /= procs
        steps = 2.0 * (procs - 1)
        self.vert_costs = (bw_to_flop * (words * steps))

    def GetVertexCosts(self):
        return self.vert_costs


class Concat():
    def __init__(self, in_ops, dim, n_procs):
        l = len(in_ops[0].out_tsr)
        n_ops = len(in_ops)

        # Concatenate the 'dim' dimension
        #self.in_tsr = (op.out_tsr for op in in_ops)
        self.dom = GetConcatenatedSize(in_ops, dim)
        self.in_tsr = self.dom
        self.out_tsr = self.dom

        # Configs
        self.dom_config_tuples = GetConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

    def GetVertexCosts(self):
        return 0


class Add():
    def __init__(self, dom_size, n_procs, pw_op_cnt=0):
        self.pw_op_cnt = pw_op_cnt

        # Domain and in / out tensors
        self.dom = tuple(dom_size)
        self.in_tsr = self.dom
        self.out_tsr = self.dom

        # configs
        self.dom_config_tuples = GetConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        # Costs
        dom_per_proc = np.ceil(self.dom / self.dom_configs)
        self.vert_costs = (1 + pw_op_cnt) * np.prod(dom_per_proc, axis = 1)

    def GetVertexCosts(self):
        return self.vert_costs


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
        self.dom_config_tuples = GetConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs[:, (m_idx, k_idx)]
        self.out_tsr_configs = self.dom_configs[:, m_idx:n_idx+1]

        # Compute the costs for configs
        self.vert_costs = ComputeGemmCost(self.dom, self.dom_configs, pw_op_cnt)

    # Returns vertex costs for different configs
    def GetVertexCosts(self):
        return self.vert_costs


class Softmax():
    def __init__(self, dom_size, n_procs):
        assert(len(dom_size) == 2)

        # Domain and input/output
        self.dom = tuple(dom_size)
        self.in_tsr = self.dom
        self.out_tsr = self.dom

        # Configs
        self.dom_config_tuples = GetConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        # Compute costs
        dom_per_proc = self.dom / self.dom_configs
        fwd_comp_costs = (exp_cost + 1) * np.prod(dom_per_proc, axis=1)
        bwd_comp_costs = 3.0 * np.prod(dom_per_proc, axis=1)
        comp_costs = fwd_comp_costs + bwd_comp_costs

        # Communication costs
        words = dom_per_proc[:, 0] # 1 word per batch => batchsize/proc
        steps = 2.0 * (n_procs - 1)
        comm_costs = bw_to_flop * (words * steps)

        self.vert_costs = comp_costs + comm_costs

    def GetVertexCosts(self):
        return self.vert_costs


class MaxPool():
    def __init__(self, fltr, stride, pad=0, img=None, n_procs=0):
        assert(len(fltr) == 2)

        self.r = fltr[0]
        self.s = fltr[1]

        self.stride = stride
        self.pad = pad

        self.stride_r, self.stride_s = MakePair(stride)
        self.pad_r, self.pad_s = MakePair(pad)

        if img:
            assert(n_procs > 1)
            assert(len(img) == 4)

            b, c, h, w = img

            h_o = int((h - self.r + 2*self.pad_r) / self.stride_r) + 1
            w_o = int((w - self.s + 2*self.pad_s) / self.stride_s) + 1

            assert(h_o > 0)
            assert(w_o > 0)

            self.in_tsr = img
            self.out_tsr = (b, c, h_o, w_o)

            self.dom = self.out_tsr
            self.dom_config_tuples = GetConfigs(self.dom, n_procs)
            self.dom_configs = np.array(self.dom_config_tuples)
            self.in_tsr_configs = self.dom_configs
            self.out_tsr_configs = self.dom_configs

    def GetVertexCosts(self):
        try:
            return self.vert_costs
        except AttributeError:
            try:
                dom_per_proc = self.dom / self.dom_configs
                costs = 3.0 * np.prod(dom_per_proc, axis=1)
                self.vert_costs = costs
            except AttributeError:
                self.vert_costs = 0

        return self.vert_costs


def AdaptivePool(h, w, img=None, n_procs=0):
    # conv_h = ((h-r+2pad)/stride)+1 = h+2, for r=pad=stride=1
    # maxpool_h = ((conv_h - maxpool_r)/ maxpool_stride) + 1
    # 1 = ((h+2) - maxpool_r) + 1, for maxpool_h=maxpool_stride=1
    # maxpool_r = h+2
    r = h + 2
    s = w + 2

    return MaxPool((r, s), 1, img=img, n_procs=n_procs)


class Conv():
    def __init__(self, img, fltr, stride, pad, n_procs, pre_maxpool=None,
            maxpool=None, pw_op_cnt=0):
        assert(len(img) == 4)
        assert(len(fltr) == 4)
        assert(img[1] == fltr[1])

        stride_r, stride_s = MakePair(stride)
        pad_r, pad_s = MakePair(pad)

        self.pre_maxpool = pre_maxpool
        self.maxpool = maxpool
        self.pw_op_cnt = pw_op_cnt

        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)
        b, c, h, w = img
        n, _, r, s = fltr

        if pre_maxpool:
            h, w = GetConvolutedSize(h, w, pre_maxpool.r, pre_maxpool.s,
                    pre_maxpool.stride, pre_maxpool.pad)
            self.pre_maxpool.dom = (b, n, h, w)

        h_o, w_o = GetConvolutedSize(h, w, r, s, stride, pad)

        # Domain
        self.dom = (b, c, h_o, w_o, r, s, n)

        # Reduced domain when maxpool is applied to 'img'. Configurations are
        # created on this reduced domain so that there is no intra-op
        # communication for max-pooling.
        if maxpool:
            h_o, w_o = GetConvolutedSize(h_o, w_o, maxpool.r, maxpool.s,
                    maxpool.stride, maxpool.pad)
            self.maxpool.dom = (b, n, h_o, w_o)

        dom_with_maxpool = (b, c, h_o, w_o, r, s, n)

        # Input/output tensors
        self.in_tsr = img
        self.out_tsr = (b, n, h_o, w_o)

        # Domain configurations
        self.dom_config_tuples = GetConfigs(dom_with_maxpool, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)

        # In/Out tensor configs
        self.in_tsr_configs = self.dom_configs[:, b_idx:w_idx+1]
        self.out_tsr_configs = self.dom_configs[:, (b_idx, n_idx, h_idx, w_idx)]
        if pre_maxpool:
            self.pre_maxpool.dom_configs = self.out_tsr_configs
        if maxpool:
            self.maxpool.dom_configs = self.out_tsr_configs

        # Get the cost for convolution op
        gemm_dom = (b * h_o * w_o, n, c * r * s)
        gemm_m = np.prod(self.dom_configs[:, (b_idx, h_idx, w_idx)],
                axis=1).reshape(-1, 1)
        gemm_n = self.dom_configs[:, n_idx].reshape(-1, 1)
        gemm_k = np.prod(self.dom_configs[:, (c_idx, r_idx, s_idx)],
                axis=1).reshape(-1, 1)
        gemm_configs = np.concatenate((gemm_m, gemm_n, gemm_k), axis=1)
        self.vert_costs = ComputeGemmCost(gemm_dom, gemm_configs, pw_op_cnt)
        assert(self.dom_configs.shape[0] == self.vert_costs.shape[0])

        # Add the cost for max-pooling
        if self.pre_maxpool:
            maxpool_costs = self.pre_maxpool.GetVertexCosts()
            assert(self.vert_costs.shape == maxpool_costs.shape)
            self.vert_costs += maxpool_costs
        if self.maxpool:
            maxpool_costs = self.maxpool.GetVertexCosts()
            assert(self.vert_costs.shape == maxpool_costs.shape)
            self.vert_costs += maxpool_costs

    def GetVertexCosts(self):
        return self.vert_costs


def AddVertex(G, op):
    node_id = G.number_of_nodes()

    print("Node: " + str(node_id) + "; Configs: " +
            str(op.dom_configs.shape[0]))

    costs = op.GetVertexCosts()
    costs = pd.Series(costs, index=op.dom_config_tuples, name='cost')

    G.add_node(node_id, op=op, costs=costs)
    return node_id


def AddEdge(G, src, tgt, src_op=None, tgt_op=None, src_tsr=None, tgt_tsr=None,
        src_cfgs=None, tgt_cfgs=None, reshape=None):
    assert(src in G)
    assert(tgt in G)

    if (not src_op) or (not tgt_op):
        node_ops = G.nodes(data='op')
        src_op = node_ops[src]
        tgt_op = node_ops[tgt]

    if src_tsr is None:
        src_tsr = src_op.out_tsr
    if tgt_tsr is None:
        tgt_tsr = tgt_op.in_tsr
    if src_cfgs is None:
        src_cfgs = src_op.out_tsr_configs
    if tgt_cfgs is None:
        tgt_cfgs = tgt_op.in_tsr_configs

    costs = GetEdgeCosts(src_tsr, tgt_tsr, src_cfgs, tgt_cfgs, reshape)
    idx = pd.MultiIndex.from_product([src_op.dom_config_tuples,
        tgt_op.dom_config_tuples], names=[str(src), str(tgt)])
    costs = pd.Series(costs, index=idx, name='cost')

    G.add_edge(src, tgt, costs=costs)


def ConcatenateVertices(G, src_verts, concat_dim, n_procs):
    assert(len(src_verts) > 1)
    node_ops = G.nodes(data='op')

    # ops and cfgs
    src_ops = []
    for v in src_verts:
        assert(v in node_ops)
        src_ops.append(node_ops[v])
    tgt_op = Concat(src_ops, concat_dim, n_procs)
    tgt_vert = AddVertex(G, tgt_op)
    tgt_tsr = tgt_op.in_tsr
    tgt_cfgs = tgt_op.dom_configs

    # Ratio of original size to concatenated size of src tensors
    sz = tgt_tsr[concat_dim]
    ratios = [float(s.out_tsr[concat_dim])/sz for s in src_ops]

    # Iterate over each src tensor and create edge and assign edge costs
    for ratio, src_vert, src_op in zip(ratios, src_verts, src_ops):
        src_tsr = src_op.out_tsr
        src_cfgs = src_op.out_tsr_configs
    
        # Get target cfgs corresponding to src tensor size based on 'ratios'
        cfgs = np.asfarray(tgt_cfgs)
        cfgs[:, concat_dim] *= ratio
        np.ceil(cfgs[:, concat_dim], out=cfgs[:, concat_dim])

        AddEdge(G, src_vert, tgt_vert, src_op, tgt_op, src_tsr, src_tsr,
                src_cfgs, cfgs)

    return tgt_vert, tgt_op


def RNNStack(G, b, h, n_layers, unroll, n_procs):
    prev_layer = []
    curr_layer = []

    def AdjustVertexCosts(node_op):
        node_op.vert_costs *= (4 * 2) # 4 for n dim and 2 for k dim

    def AdjustEdgeCosts(G, src_id, tgt_id):
        costs = nx.get_edge_attributes(G, 'costs')[(src_id, tgt_id)]
        costs *= 2

    first_layers = []
    last_layers = []
    for l in range(n_layers):
        for i in range(unroll):
            node_op = Gemm((b, h, h), n_procs, pw_op_cnt=1)
            # 8 matmults are performed in total
            AdjustVertexCosts(node_op)
            node_id = AddVertex(G, node_op)
            curr_layer.append(node_id)

            if l == 0:
                first_layers.append(node_id)
            elif l == n_layers - 1:
                last_layers.append(node_id)

            if i > 0:
                AddEdge(G, node_id - 1, node_id)
                # Both C and H matrices are transferred to next cycle
                AdjustEdgeCosts(G, node_id-1, node_id)

            if l > 0:
                # Only H matrix is transferred to next layer
                AddEdge(G, prev_layer[i], node_id)

        prev_layer, curr_layer = curr_layer, []

    return G, first_layers, last_layers


def Seq2seq(G, b, n_procs):
    n_layers = 2
    unroll = 40
    h = 1024

    vocab_size = 17191
    embed_dim = h

    # Encoder + decoder
    G, first_layers, last_layers = RNNStack(G, b, h, n_layers, 2 * unroll,
            n_procs)

    assert(len(first_layers) == 2 * unroll)
    assert(len(last_layers) == 2 * unroll)

    decoder_layers = last_layers[unroll+1:]

    # Add embeddings to inputs
    for l in first_layers:
        node_op = Embedding(b, vocab_size, embed_dim, n_procs)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id, l)

    # Add softmax to outputs of decoder
    for l in decoder_layers:
        node_op = Softmax((b, h), n_procs)
        node_id = AddVertex(G, node_op)
        AddEdge(G, l, node_id)

    return G


class PositionalEncoder():
    def __init__(self, b, vocab_size, seq_len, embed_dim, n_procs):
        self.dom = (b,)
        self.in_tsr = (b * seq_len, vocab_size)
        self.out_tsr = (b * seq_len, embed_dim)

        self.dom_config_tuples = GetConfigs(self.dom, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples).reshape(-1,1)
        rep = np.repeat(1, self.dom_configs.shape[0]).reshape(-1, 1)
        self.in_tsr_configs = np.concatenate((self.dom_configs, rep), axis=1)
        self.out_tsr_configs = self.in_tsr_configs

        dom_per_proc = np.ceil(self.dom / self.dom_configs)
        elems = np.prod(dom_per_proc, axis = 1)

        # Cost of computing and adding positional encoder to embedding matrix
        self.vert_costs = 2.0 * elems * seq_len * embed_dim
        self.vert_costs += 4.0

        # Backprop cost of synchronization
        words = elems
        procs = self.dom_configs[:,0]
        words /= procs
        steps = 2.0 * (procs - 1)
        self.vert_costs += (bw_to_flop * (words * steps))

    def GetVertexCosts(self):
        return self.vert_costs


def MultiHeadAttn(G, inp_node_id, b, seq_len, hidden_dim, n_heads, n_procs):
    # Linear layers
    node_op_q = Gemm((b * seq_len, hidden_dim, hidden_dim), n_procs)
    node_id_q = AddVertex(G, node_op_q)
    AddEdge(G, inp_node_id, node_id_q)

    node_op_k = Gemm((b * seq_len, hidden_dim, hidden_dim), n_procs)
    node_id_k = AddVertex(G, node_op_k)
    AddEdge(G, inp_node_id, node_id_k)

    node_op_v = Gemm((b * seq_len, hidden_dim, hidden_dim), n_procs)
    node_id_v = AddVertex(G, node_op_v)
    AddEdge(G, inp_node_id, node_id_v)

    d_h = hidden_dim / n_heads

    def AddAttentionEdge(G, node_op1, node_id1, node_op2, node_id2):
        src_tsr = node_op1.out_tsr
        tgt_tsr = node_op2.in_tsr

        src_cfgs = node_op1.out_tsr_configs
        tgt_cfgs = node_op2.in_tsr_configs

        src_tsr_per_proc = np.ceil(src_tsr / src_cfgs)
        tgt_tsr_per_proc = np.ceil(tgt_tsr / tgt_cfgs)

        idx = np.nonzero(src_tsr_per_proc[:, 1] >= d_h)
        src_tsr_per_proc[idx, 0] *= (src_tsr_per_proc[idx, 1] / d_h)
        src_tsr_per_proc[idx, 1] /= d_h

        costs = GetAreaNeeded(src_tsr_per_proc, tgt_tsr_per_proc, src_procs,
                tgt_procs)

        G.add_edge(node_id1, node_id2, costs=costs)

    # q * k^T - batch matmul
    node_op = Gemm((b * seq_len * n_heads, b * seq_len * n_heads, d_h), n_procs)
    node_id = AddVertex(G, node_op)
    AddAttentionEdge(G, node_op_q, node_id_q, node_op, node_id)
    AddAttentionEdge(G, node_op_k, node_id_k, node_op, node_id)

    # Softmax 
    node_op = Softmax(node_op.out_tsr, n_procs)
    node_id = AddVertex(G, node_op)
    AddEdge(node_id - 1, node_id)

    # softmax * v
    node_op = Gemm((b * seq_len * n_heads, b * seq_len * n_heads, d_h), n_procs)
    node_id = AddVertex(G, node_op)

    return node_id


def Transformer(G, b, n_procs):
    hidden_dim = 1024
    vocab_size = 17191
    seq_len = 80
    n_heads = 8

    # Positional encoder
    node_op = PositionalEncoder(b, vocab_size, seq_len, hidden_dim, n_procs)
    node_id = AddVertex(G, node_op)


def AlexNet(G, b, n_procs):
    img = (b, 3, 227, 227)

    # Conv1 + relu + maxpool + norm
    node_op = Conv(img, (96, 3, 11, 11), 4, 0, n_procs, maxpool=MaxPool((3,3), 2),
            pw_op_cnt=pw_ops_in_bn+1)
    node_id = AddVertex(G, node_op)

    # Conv2 + relu + maxpool + norm
    node_op = Conv(node_op.out_tsr, (256, 96, 5, 5), 1, 2, n_procs,
            maxpool=MaxPool((3,3), 2), pw_op_cnt=pw_ops_in_bn+1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id)

    # Conv3 + relu
    node_op = Conv(node_op.out_tsr, (384, 256, 3, 3), 1, 1, n_procs,
            pw_op_cnt=1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id)

    # Conv4 + relu
    node_op = Conv(node_op.out_tsr, (384, 384, 3, 3), 1, 1, n_procs,
            pw_op_cnt=1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id)

    # Conv5 + relu + maxpool
    node_op = Conv(node_op.out_tsr, (256, 384, 3, 3), 1, 1, n_procs,
            maxpool=MaxPool((3,3), 2), pw_op_cnt=1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id)

    # FC6 + relu
    dom = (node_op.out_tsr[0], 4096, node_op.out_tsr[1] * node_op.out_tsr[2] *
            node_op.out_tsr[3])
    node_op = Gemm(dom, n_procs, 1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id, reshape=[(0,), (1,2,3)])

    # FC7 + relu
    dom = (node_op.out_tsr[0], 4096, node_op.out_tsr[1])
    node_op = Gemm(dom, n_procs, 1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id)

    # FC8 + relu
    dom = (node_op.out_tsr[0], 1024, node_op.out_tsr[1])
    node_op = Gemm(dom, n_procs, 1)
    node_id = AddVertex(G, node_op)
    AddEdge(G, node_id - 1, node_id)

    return G


class Inception3():
    # Conv2d + BN + relu
    def AddBasicConv(self, G, in_node_id, img, in_channels, out_channels, kernel_size,
            pre_maxpool=None, maxpool=None, stride=1, padding=0):
        fltr = (out_channels, in_channels) + MakePair(kernel_size)
        node_op = Conv(img, fltr, stride, padding, self.n_procs, pw_op_cnt =
                pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)

        if in_node_id >= 0:
            AddEdge(G, in_node_id, node_id)

        return node_id, node_op

    def AddInceptionA(self, G, in_node_id, img, in_channels, pool):
        node_id1, node_op1 = self.AddBasicConv(G, in_node_id, img, in_channels, 64, 1)

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, 48, 1)
        node_id2, node_op2 = self.AddBasicConv(G, node_id, node_op.out_tsr, 48, 64, 5, padding=2)

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, 64, 1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 64, 96, 3, padding=1)
        node_id3, node_op3 = self.AddBasicConv(G, node_id, node_op.out_tsr, 96, 96, 3, padding=1)

        pre_maxpool = MaxPool((3, 3), 1, 1)
        node_id4, node_op4 = self.AddBasicConv(G, in_node_id, img, in_channels, pool, 1,
                pre_maxpool=pre_maxpool)

        node_id, node_op = ConcatenateVertices(G, (node_id1, node_id2, node_id3,
            node_id4), 1, self.n_procs)
        return node_id, node_op

    def AddInceptionB(self, G, in_node_id, img, in_channels):
        node_id1, node_op1 = self.AddBasicConv(G, in_node_id, img, in_channels, 384, 3, stride=2)

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, 64, 1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 64, 96, 3, padding=1)
        node_id2, node_op2 = self.AddBasicConv(G, node_id, node_op.out_tsr, 96, 96, 3, stride=2)

        node_op3 = MaxPool((3,3), 2, img=img, n_procs=self.n_procs)
        node_id3 = AddVertex(G, node_op3)
        AddEdge(G, in_node_id, node_id3)

        node_id, node_op = ConcatenateVertices(G, (node_id1, node_id2, node_id3), 1,
                self.n_procs)
        return node_id, node_op

    def AddInceptionC(self, G, in_node_id, img, in_channels, out_channels):
        node_id1, node_op1 = self.AddBasicConv(G, in_node_id, img, in_channels, 192, 1)

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, out_channels, 1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, out_channels,
                out_channels, (1, 7), padding=(0,3))
        node_id2, node_op2 = self.AddBasicConv(G, node_id, node_op.out_tsr, out_channels, 192,
                (7,1), padding=(3,0))

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, out_channels, 1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, out_channels,
                out_channels, (7,1), padding=(3,0))
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, out_channels,
                out_channels, (1,7), padding=(0,3))
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, out_channels,
                out_channels, (7,1), padding=(3,0))
        node_id3, node_op3 = self.AddBasicConv(G, node_id, node_op.out_tsr, out_channels, 192,
                (1,7), padding=(0,3))

        node_id4, node_op4 = self.AddBasicConv(G, in_node_id, img, in_channels, 192, 1,
                pre_maxpool=MaxPool((3, 3), 1, 1))

        node_id, node_op = ConcatenateVertices(G, (node_id1, node_id2, node_id3,
            node_id4), 1, self.n_procs)
        return node_id, node_op

    def AddInceptionD(self, G, in_node_id, img, in_channels):
        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, 192, 1)
        node_id1, node_op1 = self.AddBasicConv(G, node_id, node_op.out_tsr, 192, 320, 3, stride=2)

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, 192, 1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 192, 192, (1,7),
                padding=(0,3))
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 192, 192, (7,1),
                padding=(3,0))
        node_id2, node_op2 = self.AddBasicConv(G, node_id, node_op.out_tsr, 192, 192, 3, stride=2)

        node_op3 = MaxPool((3,3), 2, img=img, n_procs=self.n_procs)
        node_id3 = AddVertex(G, node_op3)
        AddEdge(G, in_node_id, node_id3)

        node_id, node_op = ConcatenateVertices(G, (node_id1, node_id2, node_id3), 1,
                self.n_procs)
        return node_id, node_op

    def AddInceptionE(self, G, in_node_id, img, in_channels):
        node_id1, node_op1 = self.AddBasicConv(G, in_node_id, img, in_channels, 320, 1)

        node_id2_0, node_op2_0 = self.AddBasicConv(G, in_node_id, img, in_channels, 384, 1)
        node_id2_1, node_op2_1 = self.AddBasicConv(G, node_id2_0, node_op2_0.out_tsr, 384, 384, (1,3),
                padding=(0,1))
        node_id2_2, node_op2_2 = self.AddBasicConv(G, node_id2_0, node_op2_0.out_tsr, 384, 384, (3,1),
                padding=(1,0))

        node_id, node_op = self.AddBasicConv(G, in_node_id, img, in_channels, 448, 1)
        node_id3_0, node_op3_0 = self.AddBasicConv(G, node_id, node_op.out_tsr, 448, 384,
                3, padding=1)
        node_id3_1, node_op3_1 = self.AddBasicConv(G, node_id3_0, node_op3_0.out_tsr, 384, 384, (1,3),
                padding=(0,1))
        node_id3_2, node_op3_2 = self.AddBasicConv(G, node_id3_0, node_op3_0.out_tsr, 384, 384, (3,1),
                padding=(1,0))

        node_id4, node_op4 = self.AddBasicConv(G, in_node_id, img, in_channels, 192, 1,
                pre_maxpool=MaxPool((3,3), 1, 1))

        node_id, node_op = ConcatenateVertices(G, (node_id1, node_id2_1, node_id2_2,
            node_id3_1, node_id3_2, node_id4), 1, self.n_procs)
        return node_id, node_op

    def __init__(self, G, b, n_procs):
        self.G = G
        self.n_procs = n_procs
        img = (b, 3, 299, 299)
        num_classes = 1000

        node_id, node_op = self.AddBasicConv(G, -1, img, 3, 32, 3, stride=2)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 32,
                32, 3)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 32,
                64, 3, maxpool=MaxPool((3, 3), 2), padding=1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 64,
                80, 1)
        node_id, node_op = self.AddBasicConv(G, node_id, node_op.out_tsr, 80,
                192, 3, maxpool=MaxPool((3,3), 2))

        node_id, node_op = self.AddInceptionA(G, node_id, node_op.out_tsr, 192, 32)
        node_id, node_op = self.AddInceptionA(G, node_id, node_op.out_tsr, 256, 64)
        node_id, node_op = self.AddInceptionA(G, node_id, node_op.out_tsr, 288, 64)

        node_id, node_op = self.AddInceptionB(G, node_id, node_op.out_tsr, 288)

        node_id, node_op = self.AddInceptionC(G, node_id, node_op.out_tsr, 768, 128)
        node_id, node_op = self.AddInceptionC(G, node_id, node_op.out_tsr, 768, 160)
        node_id, node_op = self.AddInceptionC(G, node_id, node_op.out_tsr, 768, 160)
        node_id, node_op = self.AddInceptionC(G, node_id, node_op.out_tsr, 768, 192)

        node_id, node_op = self.AddInceptionD(G, node_id, node_op.out_tsr, 768)

        node_id, node_op = self.AddInceptionE(G, node_id, node_op.out_tsr, 1280)
        node_id, node_op = self.AddInceptionE(G, node_id, node_op.out_tsr, 2048)

        node_op = AdaptivePool(1, 1, img=node_op.out_tsr, n_procs=self.n_procs)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)

        node_op = Gemm((node_op.out_tsr[0], num_classes, 2048), n_procs)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id, reshape=[(0,), (1, 2, 3)]) 

    def Graph(self):
        return self.G


class ResNet101():
    def AddBlock(self, img, inplanes, planes, expansion, stride, downsample):
        G = self.graph

        in_node_id = G.number_of_nodes() - 1
        identity = in_node_id

        if downsample:
            # Conv + bn + relu
            node_op = Conv(img, (planes * expansion, inplanes, 1, 1), stride, 1,
                    n_procs=self.n_procs, pw_op_cnt=pw_ops_in_bn+1)
            node_id = AddVertex(G, node_op)
            AddEdge(G, in_node_id, node_id)
            identity = node_id

        # Conv1 + bn + relu
        node_op = Conv(img, (planes, inplanes, 1, 1), 1, 1,
                n_procs=self.n_procs, pw_op_cnt=pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)
        AddEdge(G, in_node_id, node_id)
    
        # Conv2 + bn + relu
        node_op = Conv(node_op.out_tsr, (planes, planes, 3, 3), stride, 1,
                n_procs=self.n_procs, pw_op_cnt=pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)
    
        # Conv3 + bn
        node_op = Conv(node_op.out_tsr, (planes * expansion, planes, 1, 1), 1,
                1, maxpool=None, n_procs=self.n_procs,
                pw_op_cnt=pw_ops_in_bn)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)

        # Add + relu
        node_op = Add(node_op.out_tsr, self.n_procs, 1)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)
        AddEdge(G, identity, node_id)

        return node_op
    
    def AddBlocks(self, img, inplanes, planes, blocks, stride, expansion):
        G = self.graph
    
        if stride != 1 or inplanes != planes * expansion:
            downsample = True
        else:
            downsample = False
    
        node_op = self.AddBlock(img, inplanes, planes, expansion, stride,
                                downsample)
    
        inplanes = planes * expansion
        for _ in range(1, blocks):
            node_op = self.AddBlock(node_op.out_tsr, inplanes, planes,
                    expansion, stride, False)
    
        return node_op, inplanes
    
    def __init__(self, G, b, n_procs):
        self.graph = G
        self.n_procs = n_procs

        img = (b, 3, 227, 227)
        blocks = (3, 4, 23, 3)
        planes = (64, 128, 256, 512)
        num_classes = 1000
        expansion = 4
    
        # Conv1 + bn + relu + maxpool
        node_op = Conv(img, (64, 3, 7, 7), 2, 3, n_procs, maxpool=MaxPool((3,
            3), 2, 1), pw_op_cnt=pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)
    
        # Layer1
        inplanes = 64
        node_op, inplanes = self.AddBlocks(node_op.out_tsr, inplanes, planes[0],
                blocks[0], 1, expansion)
    
        # Layer2
        node_op, inplanes = self.AddBlocks(node_op.out_tsr, inplanes, planes[1],
                blocks[1], 2, expansion)
    
        # Layer3
        node_op, inplanes = self.AddBlocks(node_op.out_tsr, inplanes, planes[2],
                blocks[2], 2, expansion)
    
        # Layer4 + AdaptiveAvePooling
        node_op, inplanes = self.AddBlocks(node_op.out_tsr, inplanes, planes[3],
                blocks[3], 2, expansion)
    
        # Adaptive avgpool
        node_op = MaxPool((node_op.out_tsr[2], node_op.out_tsr[3]), 1, img =
                          node_op.out_tsr, n_procs = self.n_procs)
        assert(node_op.out_tsr[2] == 1)
        assert(node_op.out_tsr[3] == 1)
    
        # FC
        n = num_classes
        m = node_op.out_tsr[0]
        k = node_op.out_tsr[1] * node_op.out_tsr[2] * node_op.out_tsr[3]
        assert(k == 512 * expansion)
        node_op = Gemm((m, n, k), n_procs)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id, reshape=[(0,), (1,2,3)])

    def Graph(self):
        return self.graph


def TestGraph(G, batch_size, hidden_dim_size, n_procs):
    dom = [batch_size, hidden_dim_size, hidden_dim_size]

    node_id = AddVertex(G, Gemm(dom, n_procs))

    node_id = AddVertex(G, Gemm(dom, n_procs))
    AddEdge(G, node_id-1, node_id)

    node_id = AddVertex(G, Gemm(dom, n_procs))
    AddEdge(G, node_id-1, node_id)

    return G


# Creates the graph for the model
def CreateGraph(graph_type, batch_size, hidden_dim_size, n_procs):
    G = nx.DiGraph()

    if graph_type == 'test':
        G = TestGraph(G, batch_size, hidden_dim_size, n_procs)
    elif graph_type == 'alexnet':
        G = AlexNet(G, batch_size, n_procs)
    elif graph_type == 'resnet101':
        G = ResNet101(G, batch_size, n_procs).Graph()
    elif graph_type == 'inception3':
        G = Inception3(G, batch_size, n_procs).Graph()
    elif graph_type == 'seq2seq':
        G = Seq2seq(G, batch_size, n_procs)
    else:
        assert(False)

    return G

