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


# Returns edge costs for different configs. Edge cost is computed as the
# difference b/w tensor volume needed per proc by the target vertex and the tensor
# volume held per proc by the source vertex.
def GetEdgeCosts(src_op, tgt_op, reshape):
    src_tsr = src_op.out_tsr
    tgt_tsr = tgt_op.in_tsr
    src_cfgs = src_op.out_tsr_configs
    tgt_cfgs = tgt_op.in_tsr_configs

    # Calculate the domains per processor
    src_tsr_per_proc = np.ceil(src_tsr / src_cfgs)
    tgt_tsr_per_proc = np.ceil(tgt_tsr / tgt_cfgs)

    if reshape:
        tgt_tsr_per_proc = Reshape(src_tsr, tgt_tsr, src_tsr_per_proc,
                tgt_tsr_per_proc, reshape)
    else:
        assert(len(src_tsr) == len(tgt_tsr))

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
    cutoff = 32 # Minimum domain size to reduce search space
    dim = len(node_dom)

    proc_set = []
    for d in node_dom:
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


class Concat():
    def __init__(self, tensors, dim=0):
        l = len(tensors[0].out_tsr)

        # Make sure size of all dimensions except 'dim' is the same for all
        # tensors
        for i, t in enumerate(tensors):
            assert(dim < len(t.out_tsr))
            assert(len(t.out_tsr) == l)
            for j in range(l):
                if j != dim:
                    assert(t.out_tsr[j] == tensors[0].out_tsr[j])

        # Concatenate the 'dim' dimension
        self.out_tsr = list(tensors[0].out_tsr)
        for i in range(1, len(tensors)):
            self.out_tsr[dim] += tensors[i].out_tsr[dim]


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

        # Compute the costs for configs
        self.vert_costs = ComputeGemmCost(self.dom, self.dom_configs, pw_op_cnt)

    # Returns vertex costs for different configs
    def GetVertexCosts(self):
        return self.vert_costs


class MaxPool():
    def __init__(self, fltr, stride, padding=0):
        assert(len(fltr) == 2)

        self.r = fltr[0]
        self.s = fltr[1]

        self.stride_r, self.stride_s = MakePair(stride)
        self.pad_r, self.pad_s = MakePair(padding)


def AvgPool(h, w):
    # conv_h = ((h-r+2pad)/stride)+1 = h+2, for r=pad=stride=1
    # maxpool_h = ((conv_h - maxpool_r)/ maxpool_stride) + 1
    # 1 = ((h+2) - maxpool_r) + 1, for maxpool_h=maxpool_stride=1
    # maxpool_r = h+2
    r = h + 2
    s = w + 2

    return MaxPool((r, s), 1)


class Conv():
    def __init__(self, img, fltr, stride, pad, n_procs, pre_maxpool=None,
            maxpool=None, pw_op_cnt=0):
        assert(len(img) == 4)
        assert(len(fltr) == 4)
        assert(img[1] == fltr[1])

        stride_r, stride_s = MakePair(stride)
        pad_r, pad_s = MakePair(pad)

        self.maxpool = maxpool
        self.pw_op_cnt = pw_op_cnt

        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = 0, 1, 2, 3, 4, 5, 6

        b, n = img[0], fltr[0]
        c, h, w = img[1], img[2], img[3]
        r, s = fltr[2], fltr[3]

        if pre_maxpool:
            h = int((h - pre_maxpool.r + pre_maxpool.pad_r) /
                    pre_maxpool.stride_r) + 1
            w = int((w - pre_maxpool.s + pre_maxpool.pad_s) /
                    pre_maxpool.stride_s) + 1

        h_o = int((h - r + 2*pad_r) / stride_r) + 1
        w_o = int((w - s + 2*pad_s) / stride_s) + 1

        # Domain
        self.dom = (b, c, h_o, w_o, r, s, n)

        # Reduced domain when maxpool is applied to 'img'. Configurations are
        # created on this reduced domain so that there is no intra-op
        # communication for max-pooling.
        if maxpool:
            h_o = int((h_o - maxpool.r + maxpool.pad_r) / maxpool.stride_r) + 1
            w_o = int((w_o - maxpool.s + maxpool.pad_s) / maxpool.stride_s) + 1
            self.maxpool.dom = (b, n, h_o, w_o)

        dom_with_maxpool = (b, c, h_o, w_o, r, s, n)

        # Input/output tensors
        self.in_tsr = img
        self.out_tsr = (b, n, h_o, w_o)

        # Domain configurations
        self.dom_config_tuples = GetNodeConfigs(dom_with_maxpool, n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)

        # In/Out tensor configs
        self.in_tsr_configs = self.dom_configs[:, b_idx:w_idx+1]
        self.out_tsr_configs = self.dom_configs[:, (b_idx, n_idx, h_idx, w_idx)]
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
        if self.maxpool:
            dom_per_proc = self.maxpool.dom / self.maxpool.dom_configs
            maxpool_costs = 3.0 * np.prod(dom_per_proc, axis=1)

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


def AddEdge(G, src, tgt, src_op=None, tgt_op=None, reshape=None):
    assert(src in G)
    assert(tgt in G)

    if (not src_op) or (not tgt_op):
        node_ops = G.nodes(data='op')
        src_op = node_ops[src]
        tgt_op = node_ops[tgt]

    costs = GetEdgeCosts(src_op, tgt_op, reshape)
    idx = pd.MultiIndex.from_product([src_op.dom_config_tuples,
        tgt_op.dom_config_tuples], names=[str(src), str(tgt)])
    costs = pd.Series(costs, index=idx, name='cost')

    G.add_edge(src, tgt, costs=costs)


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
    def AddBasicConv(self, G, img, in_channels, out_channels, kernel_size,
            pre_maxpool=None, maxpool=None, stride=1, padding=0):
        fltr = (out_channels, in_channels) + MakePair(kernel_size)
        node_op = Conv(img, fltr, stride, padding, self.n_procs, pw_op_cnt =
                pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)

        return node_op

    def AddInceptionA(self, G, img, in_channels, pool):
        node_op1 = self.AddBasicConv(G, img, in_channels, 64, 1)

        node_op = self.AddBasicConv(G, img, in_channels, 48, 1)
        node_op2 = self.AddBasicConv(G, node_op.out_tsr, 48, 64, 5, padding=2)

        node_op = self.AddBasicConv(G, img, in_channels, 64, 1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 64, 96, 3, padding=1)
        node_op3 = self.AddBasicConv(G, node_op.out_tsr, 96, 96, 3, padding=1)

        pre_maxpool = Maxpool((3, 3), 1, 1)
        node_op4 = self.AddBasicConv(G, img, in_channels, pool, 1,
                pre_maxpool=pre_maxpool)

        # TODO
        node_op = Concat((node_op1, node_op2, node_op3, node_op4), 1)
        return node_op

    def AddInceptionB(self, G, img, in_channels):
        node_op1 = self.AddBasicConv(G, img, in_channels, 384, 3, stride=2)

        node_op = self.AddBasicConv(G, img, in_channels, 64, 1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 64, 96, 3, padding=1)
        node_op2 = self.AddBasicConv(G, node_op.out_tsr, 96, 96, 3,
                maxpool=Maxpool((3,3), 2), stride=2)

        #TODO
        node_op = Concat((node_op1, node_op2, node_op3), 1)
        return node_op

    def AddInceptionC(self, G, img, in_channels, out_channels):
        node_op1 = self.AddBasicConv(G, img, in_channels, 192, 1)

        node_op = self.AddBasicConv(G, img, in_channels, out_channels, 1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, out_channels,
                out_channels, (1, 7), padding=(0,3))
        node_op2 = self.AddBasicConv(G, node_op.out_tsr, out_channels, 192,
                (7,1), padding=(3,0))

        node_op = self.AddBasicConv(G, img, in_channels, out_channels, 1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, out_channels,
                out_channels, (7,1), padding=(3,0))
        node_op = self.AddBasicConv(G, node_op.out_tsr, out_channels,
                out_channels, (1,7), padding=(0,3))
        node_op = self.AddBasicConv(G, node_op.out_tsr, out_channels,
                out_channels, (7,1), padding=(3,0))
        node_op3 = self.AddBasicConv(G, node_op.out_tsr, out_channels, 192,
                (1,7), padding=(0,3))

        node_op4 = self.AddBasicConv(G, img, in_channels, 192, 1,
                pre_maxpool=MaxPool((3, 3), 1, 1))

        # TODO
        node_op = Concat((node_op1, node_op2, node_op3, node_op4), 1)
        return node_op

    def AddInceptionD(self, G, img, in_channels):
        node_op = self.AddBasicConv(G, img, in_channels, 192, 1)
        node_op1 = self.AddBasicConv(G, node_op.out_tsr, 192, 320, 3, stride=2)

        node_op = self.AddBasicConv(G, img, in_channels, 192, 1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 192, 192, (1,7),
                padding=(0,3))
        node_op = self.AddBasicConv(G, node_op.out_tsr, 192, 192, (7,1),
                padding=(3,0))
        node_op2 = self.AddBasicConv(G, node_op.out_tsr, 192, 192, 3, stride=2)

        node_op3 = MaxPool(G, img, (3,3), 2)

        # TODO
        node_op = Concat((node_op1, node_op2, node_op3), 1)
        return node_op

    def AddInceptionE(self, G, img, in_channels):
        node_op1 = self.AddBasicConv(G, img, in_channels, 320, 1)

        node_op2_0 = self.AddBasicConv(G, img, in_channels, 384, 1)
        node_op2_1 = self.AddBasicConv(G, node_op2_0.out_tsr, 384, 384, (1,3),
                padding=(0,1))
        node_op2_2 = self.AddBasicConv(G, node_op2_0.out_tsr, 384, 384, (3,1),
                padding=(1,0))
        node_op2 = Concat((node_op2_1, node_op2_2), 1)

        node_op = self.AddBasicConv(G, img, in_channels, 448, 1)
        node_op3_0 = self.AddBasicConv(G, node_op.out_tsr, 448, 384, 3, 1)
        node_op3_1 = self.AddBasicConv(G, node_op3_0.out_tsr, 384, 384, (1,3),
                padding=(0,1))
        node_op3_2 = self.AddBasicConv(G, node_op3_0.out_tsr, 384, 384, (3,1),
                padding=(1,0))
        node_op3 = Concat((node_op3_1, node_op3_2), 1)

        node_op4 = self.AddBasicConv(G, img, in_channels, 192, 1,
                pre_maxpool=MaxPool((3,3), 1, 1))

        node_op = Concat((node_op1, node_op2, node_op3, node_op4), 1)
        return node_op

    def __init__(self, G, b, n_procs):
        self.n_procs = n_procs
        img = (b, 3, 299, 299)
        num_classes = 1000

        node_op = self.AddBasicConv(G, img, 3, 32, 3, stride=2)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 32, 32, 3)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 32, 64, 3,
                maxpool=MaxPool((3, 3), 2), padding=1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 64, 80, 1)
        node_op = self.AddBasicConv(G, node_op.out_tsr, 80, 192, 3,
                maxpool=MaxPool((3,3), 2))

        node_op = self.AddInceptionA(G, node_op.out_tsr, 192, 32)
        node_op = self.AddInceptionA(G, node_op.out_tsr, 256, 64)
        node_op = self.AddInceptionA(G, node_op.out_tsr, 288, 64)

        node_op = self.AddInceptionB(G, node_op.out_tsr, 288)

        node_op = self.AddInceptionC(G, node_op.out_tsr, 768, 128)
        node_op = self.AddInceptionC(G, node_op.out_tsr, 768, 160)
        node_op = self.AddInceptionC(G, node_op.out_tsr, 768, 160)
        node_op = self.AddInceptionC(G, node_op.out_tsr, 768, 192)

        node_op = self.AddInceptionD(G, node_op.out_tsr, 768)

        node_op = self.AddInceptionE(G, node_op.out_tsr, 1280)
        node_op = self.AddInceptionE(G, node_op.out_tsr, 2048, avgpool=True)

        node_op = Gemm((node_op.out_tsr[0], num_classes, 2048), n_procs)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id, reshape=[(0,), (1, 2, 3)]) 



class ResNet101():
    def AddBlock(self, img, inplanes, planes, expansion, stride, downsample, avgpool):
        G = self.graph

        # Conv1 + bn + relu
        node_op = Conv(img, (planes, inplanes, 1, 1), 1, 1,
                n_procs=self.n_procs, pw_op_cnt=pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)
        in_node_id = node_id-1
    
        # Conv2 + bn + relu
        node_op = Conv(node_op.out_tsr, (planes, planes, 3, 3), stride, 1,
                n_procs=self.n_procs, pw_op_cnt=pw_ops_in_bn+1)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)
    
        # Aggregated avgpool
        if avgpool:
            assert(downsample == False) # If downsample is true, then avgpool
                                        # has to be fused with the conv within
                                        # downsample
            maxpool = AvgPool(node_op.out_tsr[2], node_op.out_tsr[3])
        else:
            maxpool = None

        # Conv3 + bn
        node_op = Conv(node_op.out_tsr, (planes * expansion, planes, 1, 1), 1,
                1, maxpool=maxpool, n_procs=self.n_procs,
                pw_op_cnt=pw_ops_in_bn)
        node_id = AddVertex(G, node_op)
        AddEdge(G, node_id-1, node_id)

        if avgpool:
            assert(node_op.out_tsr[2] == 1)
            assert(node_op.out_tsr[3] == 1)
    
        if downsample:
            # Conv + bn + relu
            node_op = Conv(img, (planes * expansion, inplanes, 1, 1), stride, 1,
                    n_procs=self.n_procs, pw_op_cnt=pw_ops_in_bn+1)
            node_id = AddVertex(G, node_op)
            AddEdge(G, node_id-1, node_id)
            AddEdge(G, in_node_id, node_id)
    
        return node_op
    
    def AddBlocks(self, img, inplanes, planes, blocks, stride, expansion, avgpool=False):
        G = self.graph
    
        if stride != 1 or inplanes != planes * expansion:
            downsample = True
        else:
            downsample = False
    
        node_op = self.AddBlock(img, inplanes, planes, expansion, stride,
                downsample, False)
    
        inplanes = planes * expansion
        for _ in range(1, blocks-1):
            node_op = self.AddBlock(node_op.out_tsr, inplanes, planes,
                    expansion, stride, False, False)
        assert(blocks > 0) # If blocks==0, then avgpool has to be fused with the
                           # initial block.
        node_op = self.AddBlock(node_op.out_tsr, inplanes, planes,
                expansion, stride, False, avgpool)
    
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
                blocks[3], 2, expansion, True)
    
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
    else:
        assert(False)

    return G

