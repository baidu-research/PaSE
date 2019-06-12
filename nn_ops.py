import networkx as nx
import numpy as np
import pandas as pd

import itertools
from functools import reduce
import operator as op


def prod(v):
    return reduce(op.mul, v, 1)


class Tensor(tuple):
    def SetAsInput(self):
        self.is_input = True


def BytesToFlops(bytes):
    try:
        return BytesToFlops.bw_to_flops * bytes
    except AttributeError:
        p100_peak_flop = float(10.6 * 1000) # GFLOPs
        #p100_bw = float((36.72 * 2) / 8) # NVLink Unidirectional for 2 sublinks per direction.
        #                                 # GBytes/sec = b/8 GWords/sec
        p100_bw = float(13.0 / 8.0) # PCIe bidirectional GWords / sec
        
        v100_peak_flop = float(15.7 * 1000) # GFLOPs
        v100_bw = float((47.99 * 3) / 8) # Unidirectional for 3 sublinks per direction.
                                         # GBytes/sec = b/8 GWords/sec
        
        peak_flop = v100_peak_flop
        bw = v100_bw
        BytesToFlops.bw_to_flops = float(peak_flop / bw)

        return BytesToFlops.bw_to_flops * bytes


# Returns a list of factors of a number 'n'
def factors(n):
    assert(n > 0)
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) +
        1) if n % i == 0)))


# Converts 'v' into a tuple (v, v) if 'v' is a scalar
def MakePair(v):
    if hasattr(v, "__len__"):
        assert len(v) == 2
        return v
    else:
        return (v, v)


# Generates list of configurations for an operation
# cutoff - Minimum domain size to reduce search space
def GetConfigs(dom, n_procs, cutoff = 4):
    dim = len(dom)

    proc_set = []
    for d in dom:
        s = factors(d)
        l = [e for e in s if d/e >= cutoff]
        if len(l) <= 0:
            l = [1]
        proc_set.append(l)

    configs = [c for c in itertools.product(*proc_set) if prod(c) <= n_procs]
    return configs


def GetAllReduceCost(words, procs):
    # All-reduce cost = 2*((n*k)/P)*(P-1)
    chunks = words / procs # The elements are split into 'procs' chunks
    steps = 2.0 * (procs - 1)
    costs = BytesToFlops(words * steps) # When procs = 1, the cost is 0

    return costs


def ComputeGemmCosts(dom, dom_configs, pw_op_cnt, trainable=True):
    m_idx, n_idx, k_idx = 0, 1, 2
    dom_per_proc = dom / dom_configs

    # Cost for 1 GEMM in fwd phase + 2 GEMMs in bwd phase
    costs = 3.0 * np.prod(dom_per_proc, axis=1)

    # Cost of pointwise op
    if pw_op_cnt > 0:
        pw_cost = dom_per_proc[:, m_idx] * dom_per_proc[:, n_idx]
        costs += pw_op_cnt * 3 * pw_cost # Factor 3 is to
                                         # account for 1 pointwise op (per
                                         # element) in fwd phase, 1
                                         # differentiation op in bwd phase, and
                                         # 1 hadamard product in bwd phase
    
    # Cost for reducing the output during fwd phase
    words = np.prod(dom_per_proc[:, m_idx:n_idx+1], axis=1)
    costs += GetAllReduceCost(words, dom_configs[:, k_idx])
    
    if trainable:
        weights_per_proc = dom_per_proc[:, k_idx] * dom_per_proc[:, n_idx]

        # Matrix addition cost for weight update
        costs += weights_per_proc
        # Cost for gradient update during bwd phase
        costs += GetAllReduceCost(weights_per_proc, dom_configs[:, m_idx])

    return costs


# Ghost communication costs for convolution, pooling
def ComputeGhostCommCosts(tsr, configs, r, s):
    assert len(tsr) == configs.shape[1] == 4

    b_idx, c_idx, h_idx, w_idx = range(4)

    tsr_per_proc = tsr / configs
    tsr_per_proc_with_ghosts = tsr_per_proc[:, h_idx:w_idx+1]

    # Add ghost elements along h and w dims if the dimension is split among more
    # than one proc
    np.add(tsr_per_proc_with_ghosts[:, 0], r, where=(configs[:, h_idx] > 1),
            out=tsr_per_proc_with_ghosts[:, 0])
    np.add(tsr_per_proc_with_ghosts[:, 1], s, where=(configs[:, w_idx] > 1),
            out=tsr_per_proc_with_ghosts[:, 1])

    # Get the ghost element count
    inner_elems = np.prod(tsr_per_proc[:, h_idx:w_idx+1], axis=1)
    outer_elems = np.prod(tsr_per_proc_with_ghosts, axis=1)
    ghost_elems = outer_elems - inner_elems

    # Multiply it by other dimensions
    ghost_elems *= tsr_per_proc[:, b_idx]
    ghost_elems *= tsr_per_proc[:, c_idx]

    costs = BytesToFlops(ghost_elems)
    return costs


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


# Returns edge costs for different configs. Edge cost is computed as the
# difference b/w tensor volume needed per proc by the target vertex and the tensor
# volume held per proc by the source vertex.
def GetEdgeCosts(tsr, src_cfgs, tgt_cfgs):
    # Calculate the domains per processor
    src_tsr_per_proc = tsr / src_cfgs
    tgt_tsr_per_proc = tsr / tgt_cfgs

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
    costs = BytesToFlops(costs)

    return costs


def GetConvolutedSize(h, w, r, s, stride, pad):
    stride_r, stride_s = MakePair(stride)
    pad_r, pad_s = MakePair(pad)

    h_o = int((h - r + 2*pad_r) / stride_r) + 1
    w_o = int((w - s + 2*pad_s) / stride_s) + 1

    return h_o, w_o


# Parent operator class
class Ops():
    # Static variables
    G = None # Default graph
    default_procs = 0 # Can be set once and reused for the entire graph.
    tsr_to_node_id = {}

    def __init__(self, dom, in_tsrs, out_tsrs, n_procs):
        has_right_type = lambda t: isinstance(t, Tensor) or all(isinstance(e,
            Tensor) for e in t)

        n_procs = n_procs or Ops.default_procs

        assert n_procs > 0
        assert has_right_type(in_tsrs)
        assert has_right_type(out_tsrs)

        self.dom = tuple(dom)
        self.in_tsrs = in_tsrs
        self.out_tsrs = out_tsrs
        self.n_procs = n_procs

        regularize_tsrs = lambda x: (x,) if isinstance(x, Tensor) else x
        regularize_configs = lambda x: (x,) if isinstance(x, np.ndarray) else x

        self.in_tsrs = regularize_tsrs(self.in_tsrs)
        self.out_tsrs = regularize_tsrs(self.out_tsrs)
        self.out_tsrs = tuple(Tensor(t) for t in self.out_tsrs) # Make sure out_tsrs
                                                                # are fresh copies

        self.in_tsrs_cnt = len(self.in_tsrs)
        self.out_tsrs_cnt = len(self.out_tsrs)

        self.ComputeCosts()
        self.in_tsr_configs = regularize_configs(self.in_tsr_configs)
        self.out_tsr_configs = regularize_configs(self.out_tsr_configs)

        assert len(self.in_tsr_configs) == self.in_tsrs_cnt
        assert len(self.out_tsr_configs) == self.out_tsrs_cnt

        def AddVertex():
            if Ops.G is None:
                Ops.G = nx.DiGraph()
            node_id = Ops.G.number_of_nodes()
        
            print("Node: " + str(node_id) + "; Type: " +
                    str(self.__class__.__name__) + "; Configs: " +
                    str(self.dom_configs.shape[0]))
        
            costs = self.GetCosts()
            costs = pd.Series(costs, index=self.dom_config_tuples, name='cost')
        
            Ops.G.add_node(node_id, op=self, costs=costs)
            self.node_id = node_id

            for i, t in enumerate(self.out_tsrs):
                Ops.tsr_to_node_id[id(t)] = (node_id, i)

        def AddEdge(tsr, idx):
            try:
                src, src_tsr_idx = Ops.tsr_to_node_id[id(tsr)]
            except KeyError:
                # Source is an input tensor
                assert tsr.is_input == True
                return

            tgt = self.node_id
            tgt_tsr_idx = idx

            assert src in Ops.G
            assert tgt in Ops.G
        
            node_ops = Ops.G.nodes(data='op')
            src_op = node_ops[src]
            tgt_op = self
        
            src_cfgs = src_op.GetOutTensorConfigs(src_tsr_idx)
            tgt_cfgs = tgt_op.GetInTensorConfigs(tgt_tsr_idx)
        
            costs = GetEdgeCosts(tsr, src_cfgs, tgt_cfgs)
            idx = pd.MultiIndex.from_product([src_op.dom_config_tuples,
                tgt_op.dom_config_tuples], names=[str(src), str(tgt)])
            costs = pd.Series(costs, index=idx, name='cost')
        
            Ops.G.add_edge(src, tgt, costs=costs)

        # Add a vertex to the graph for the current op
        AddVertex()

        # Add edges to the predecessors
        for i, t in enumerate(self.in_tsrs):
            AddEdge(t, i)


    def ComputeCosts(self):
        self.dom_config_tuples = GetConfigs(self.dom, self.n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        assert self.dom_configs.ndim == 2

        self.in_tsr_configs = None
        self.out_tsr_configs = None
        self.costs = 0

    def GetInTensor(self, idx):
        assert idx < self.in_tsrs_cnt
        return self.in_tsrs[idx]

    def GetOutTensor(self, idx):
        assert idx < self.out_tsrs_cnt
        return self.out_tsrs[idx]

    def GetInTensorConfigs(self, idx):
        assert idx < self.in_tsrs_cnt
        return self.in_tsr_configs[idx]

    def GetOutTensorConfigs(self, idx):
        assert idx < self.out_tsrs_cnt
        return self.out_tsr_configs[idx]

    def GetInTensorConfig(self, idx, dom_config):
        loc = np.where(self.dom_configs == dom_config)[0]
        assert len(loc) >= 1

        cfgs = self.GetInTensorConfigs(idx)
        return cfgs[loc[0]]

    def GetOutTensorConfig(self, idx, dom_config):
        loc = np.where(self.dom_configs == dom_config)[0]
        assert len(loc) >= 1

        cfgs = self.GetOutTensorConfigs(idx)
        return cfgs[loc[0]]

    # Returns vertex costs for different configs
    def GetCosts(self):
        return self.costs


# Elementwise ops such as add, mul, etc.,
class Elementwise(Ops):
    def __init__(self, tsr1, tsr2, n_procs=None, pw_op_cnt=0):
        # Both the inputs should have same rank and shape
        assert len(tsr1) == len(tsr2)
        assert all(t1 == t2 for t1, t2 in zip(tsr1, tsr2))

        self.pw_op_cnt = pw_op_cnt
        out_tsr = Tensor(tsr1)
        super().__init__(tsr1, (tsr1, tsr2), out_tsr, n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()
        self.in_tsr_configs = (self.dom_configs, self.dom_configs)
        self.out_tsr_configs = self.dom_configs

        dom_per_proc = np.prod(self.dom / self.dom_configs, axis=1)
        self.costs = (1 + self.pw_op_cnt) * dom_per_proc


class Ravel(Ops):
    def __init__(self, tsr, n_procs=None):
        out_tsr = Tensor((prod(tsr), ))
        super().__init__(tsr, tsr, out_tsr, n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = np.prod(self.dom_configs, axis=1, keepdims=True)

        tsr_len = len(self.GetInTensor(0))
        ravelled_configs = np.empty_like(self.in_tsr_configs)
        cfgs = np.copy(self.out_tsr_configs)
        for i in range(tsr_len):
            np.minimum(cfgs[:,0], self.GetInTensor(0)[i],
                    out=ravelled_configs[:,i])
            cfgs[:,0] //= ravelled_configs[:,i]

        src_tsr_per_proc = self.GetInTensor(0) / self.in_tsr_configs
        tgt_tsr_per_proc = self.GetInTensor(0) / ravelled_configs

        src_procs = np.prod(self.in_tsr_configs, axis=1)
        tgt_procs = self.out_tsr_configs[:,0]

        area_needed = GetAreaNeeded(src_tsr_per_proc, tgt_tsr_per_proc,
                src_procs, tgt_procs)
        costs = 2.0 * np.where(area_needed < 0, 0, area_needed)
        self.costs = BytesToFlops(costs)


class Unravel(Ops):
    def __init__(self, tsr, shape, n_procs=None):
        # Input should be a flattened array
        assert len(tsr) == 1
        super().__init__(shape, tsr, Tensor(shape), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()
        self.in_tsr_configs = np.prod(self.dom_configs, axis=1, keepdims=True)
        self.out_tsr_configs = self.dom_configs

        tsr_len = len(self.GetOutTensor(0))
        unravelled_configs = np.empty_like(self.out_tsr_configs)
        cfgs = np.copy(self.in_tsr_configs)
        for i in range(tsr_len):
            np.minimum(cfgs[:,0], self.GetOutTensor(0)[i],
                    out=unravelled_configs[:,i])
            cfgs[:,0] //= unravelled_configs[:,i]

        src_tsr_per_proc = self.GetOutTensor(0) / unravelled_configs
        tgt_tsr_per_proc = self.GetOutTensor(0) / self.out_tsr_configs

        src_procs = self.in_tsr_configs[:,0]
        tgt_procs = np.prod(self.out_tsr_configs, axis=1)

        area_needed = GetAreaNeeded(src_tsr_per_proc, tgt_tsr_per_proc,
                src_procs, tgt_procs)
        costs = 2.0 * np.where(area_needed < 0, 0, area_needed)
        self.costs = BytesToFlops(costs)


def Reshape(tsr, shape, n_procs=None):
    ravel = Ravel(tsr, n_procs)
    unravel = Unravel(ravel.GetOutTensor(0), shape, n_procs)
    return unravel


class Transpose(Ops):
    def __init__(self, in_tsr, perm, n_procs=None):
        assert len(in_tsr) == len(perm)
        self.perm = perm

        out_tsr = Tensor(tuple(in_tsr[p] for p in perm))
        super().__init__(in_tsr, in_tsr, out_tsr, n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs[:, tuple(p for p in self.perm)]
        self.costs = 0


# Fully connected layer
class FC(Ops):
    def __init__(self, in_tsr, n_units, n_procs=None, pw_op_cnt=0):
        assert len(in_tsr) == 2

        self.pw_op_cnt = pw_op_cnt

        m_idx, n_idx, k_idx = range(3)

        # Domain and input/output tensors
        dom = (in_tsr[0], n_units, in_tsr[1])
        out_tsr = Tensor((dom[m_idx], dom[n_idx]))
        super().__init__(dom, in_tsr, out_tsr, n_procs)

    def ComputeCosts(self):
        m_idx, n_idx, k_idx = range(3)

        # Configurations
        super().ComputeCosts()
        self.in_tsr_configs = self.dom_configs[:, (m_idx, k_idx)]
        self.out_tsr_configs = self.dom_configs[:, m_idx:n_idx+1]

        # Compute the costs for configs
        self.costs = ComputeGemmCosts(self.dom, self.dom_configs, self.pw_op_cnt)


# Batched matmul
class MatMul(Ops):
    def __init__(self, tsr1, tsr2, n_procs=None, pw_op_cnt=0):
        # Both tensors should be of same rank and >=2, inner most two dimensions
        # correspond to valid GEMM, and outer dimensions should match.
        assert len(tsr1) == len(tsr2) >= 2
        assert tsr1[-1] == tsr2[-2]
        assert all(t1 == t2 for t1, t2 in zip(tsr1[:-2], tsr2[:-2]))
        self.pw_op_cnt = pw_op_cnt

        dom = tsr1[:-1] + (tsr2[-1], tsr2[-2])
        in_tsrs = (tsr1, tsr2)
        out_tsr = Tensor(dom[:-1])
        super().__init__(dom, in_tsrs, out_tsr, n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        # Configurations
        m_idx, n_idx, k_idx = -3, -2, -1
        batch_dims = list(range(self.dom_configs.shape[1] - 3))
        self.in_tsr_configs = (self.dom_configs[:, batch_dims + [m_idx, k_idx]],
                self.dom_configs[:, batch_dims + [k_idx, n_idx]])
        self.out_tsr_configs = self.dom_configs[:, :-1]

        # Compute the cost for a single GEMM, and multiply it with the number of
        # batches per proc
        gemm_dom = self.dom[-3:]
        gemm_dom_configs = self.dom_configs[:,-3:]
        self.costs = ComputeGemmCosts(gemm_dom, gemm_dom_configs,
                self.pw_op_cnt, trainable=False)
        batches_per_proc = np.prod(self.dom[:-3] / self.dom_configs[:,:-3],
                axis=1)
        assert batches_per_proc.shape == self.costs.shape
        self.costs *= batches_per_proc


'''
def Einsum(eq, tsr1, tsr2, n_procs=None, pw_op_cnt=0):
    tsr1_dims, rem_dims = eq.split(',')
    tsr2_dims, out_dims = rem_dims.split('->')

    tsr1_dim_set = set(tsr1_dims)
    tsr2_dim_set = set(tsr2_dims)
    out_dim_set = set(out_dims)
    red_dim_set = (tsr1_dim_set | tsr2_dim_set) - out_dim_set

    dim1 = (tsr1_dim_set - red_dim_set)
    dim2 = (tsr2_dim_set - red_dim_set)
    tsr1_batch_set = dim1 & tsr2_dim_set
    tsr2_batch_set = dim2 & tsr2_dim_set

    m_dim = dim1 - tsr1_batch_set
    n_dim = dim2 - tsr2_batch_set

    assert len(m_dim) <= 1
    assert len(n_dim) <= 1

    tsr1_batch_dims = [d for d in tsr1_dims if d in tsr1_batch_set]
    tsr2_batch_dims = [d for d in tsr2_dims if d in tsr2_batch_set]

    assert len(tsr1_dims) == len(tsr1)
    assert len(tsr2_dims) == len(tsr2)

    tsr1_red_axis = [i for i, d in enumerate(tsr1_dims) if d in red_dims]
    tsr2_red_axis = [i for i, d in enumerate(tsr2_dims) if d in red_dims]

    assert len(tsr1_red_axis) == len(tsr2_red_axis)

    assert all(tsr1[i] == tsr2[i] for 

    assert all(a < len(tsr1) and a < len(tsr2) for a in reduction_axis)
'''


# Convolution
class Conv(Ops):
    def __init__(self, img, fltr, stride=1, pad=0, n_procs=None, pw_op_cnt=0):
        assert len(img) == 4
        assert len(fltr) == 4
        assert img[1] == fltr[1]

        self.pw_op_cnt = pw_op_cnt

        b, c, h, w = img
        n, _, r, s = fltr
        h_o, w_o = GetConvolutedSize(h, w, r, s, stride, pad)

        # Domain
        dom = (b, c, h_o, w_o, r, s, n)
        out_tsr = Tensor((b, n, h_o, w_o))
        super().__init__(dom, img, out_tsr, n_procs)

    def ConvertToGemmDom(self):
        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)
        b, c, h_o, w_o, r, s, n = self.dom

        gemm_dom = (b * h_o * w_o, n, c * r * s)
        gemm_m = np.prod(self.dom_configs[:, (b_idx, h_idx, w_idx)],
                axis=1, keepdims=True)
        gemm_n = self.dom_configs[:, n_idx].reshape(-1, 1)
        gemm_k = np.prod(self.dom_configs[:, (c_idx, r_idx, s_idx)],
                axis=1, keepdims=True)
        gemm_configs = np.concatenate((gemm_m, gemm_n, gemm_k), axis=1)

        return gemm_dom, gemm_configs

    '''
    # Fuse pooling
    def Fuse(self, pool):
        assert isinstance(pool, Pooling)
        # Make sure we haven't computed configs yet
        assert not hasattr(self, 'dom_configs')

        try:
            pool_dom_configs = pool.dom_configs
        except AttributeError:
            pool.ComputeCosts()
            pool_dom_configs = pool.dom_configs

        # Compute original configs
        dom_config_tuples = GetConfigs(self.dom, self.n_procs)
        dom_configs = np.array(dom_config_tuples)

        # Get configs that intersect with pool's configs
        dom_configs = dom_configs[np.all(np.isin(dom_configs[:,:4],
            pool_dom_configs), axis=1), :]

        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)
        self.dom_configs = dom_configs
        self.dom_config_tuples = [tuple(e) for e in dom_configs]
        self.in_tsr_configs = self.dom_configs[:, b_idx:w_idx+1]
        self.out_tsr_configs = self.dom_configs[:, (b_idx, n_idx, h_idx, w_idx)]

        # Compute costs
        gemm_dom, gemm_configs = self.ConvertToGemmDom()
        self.costs = ComputeGemmCosts(gemm_dom, gemm_configs, self.pw_op_cnt)

        # Add pooling costs
        assert self.costs.shape == pool.costs.shape
        self.costs += pool.costs
        self.out_tsrs = pool.out_tsrs
    '''

    def ComputeCosts(self):
        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)

        # Configurations
        super().ComputeCosts()
        self.in_tsr_configs = self.dom_configs[:, b_idx:w_idx+1]
        self.out_tsr_configs = self.dom_configs[:, (b_idx, n_idx, h_idx, w_idx)]

        # Represent convolution as GEMM computation, and compute the cost for
        # GEMM op
        gemm_dom, gemm_configs = self.ConvertToGemmDom()
        self.costs = ComputeGemmCosts(gemm_dom, gemm_configs, self.pw_op_cnt)

        # Add costs for ghost communications
        self.costs += ComputeGhostCommCosts(self.GetInTensor(0),
                self.in_tsr_configs, self.dom[r_idx], self.dom[s_idx])


# Pooling - Maxpool, Avgpool
class Pooling(Ops):
    def __init__(self, img, fltr, stride=1, pad=0, n_procs=None):
        assert len(img) == 4
        assert len(fltr) == 2

        b, c, h, w = img
        self.r, self.s = fltr
        h_o, w_o = GetConvolutedSize(h, w, self.r, self.s, stride, pad)

        dom = (b, c, h_o, w_o)
        super().__init__(dom, img, Tensor(dom), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        dom_per_proc = self.dom / self.dom_configs
        self.costs = np.prod(dom_per_proc, axis=1)

        # Add costs for ghost communications
        self.costs += ComputeGhostCommCosts(self.GetInTensor(0),
                self.in_tsr_configs, self.r, self.s)


class Concat(Ops):
    def __init__(self, in_tsrs, axis, n_procs=None):
        tsr0 = in_tsrs[0]
        rank = len(tsr0)
        self.axis = axis

        assert len(in_tsrs) >= 2
        # All tensors should be of same rank, and concat axis should be valid
        assert(axis < rank)
        assert all(len(t) == rank for t in in_tsrs)
        # All tensors should have same dimensions along non-concatenated axes
        assert all(t[i] == tsr0[i] for t in in_tsrs for i in range(rank) if i !=
                axis)

        concatenated_size = reduce(op.add, (t[axis] for t in in_tsrs))
        dom = list(tsr0)
        dom[axis] = concatenated_size
        in_tsrs = tuple(t for t in in_tsrs)
        super().__init__(dom, in_tsrs, Tensor(dom), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        # Remove configs where distribution of concatenated dimension doesn't
        # align with the tensor boundaries
        concat_axis_sizes = list(t[self.axis] for t in self.in_tsrs)
        valid_idx = np.all(list(np.mod(s, self.dom_configs[:, self.axis]) == 0 for s
            in concat_axis_sizes), axis=0)
        self.dom_configs = self.dom_configs[valid_idx, :]
        self.dom_config_tuples = [tuple(e) for e in self.dom_configs]

        self.in_tsr_configs = (self.dom_configs,) * len(self.in_tsrs)
        self.out_tsr_configs = self.dom_configs
        self.costs = 0


class Norm(Ops):
    def __init__(self, in_tsr, axis, n_procs=None):
        assert len(in_tsr) > 1
        assert axis < len(in_tsr)
        self.axis = axis
        super().__init__(in_tsr, in_tsr, Tensor(in_tsr), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        # Computation cost for fwd phase. Same cost for bwd phase too.
        dom_per_proc = self.dom / self.dom_configs
        elems = np.prod(dom_per_proc, axis=1)
        self.costs = (2 * 7.0) * elems

        # Communication cost for fwd phase: Reduction and broadcast of mean and
        # variance. 2 reductions for fwd phase, and 4 for bwd phase.
        self.costs += 6.0 * GetAllReduceCost(elems / dom_per_proc[:, self.axis],
                self.dom_configs[:, self.axis])


def BatchNorm(in_tsr, n_procs=None):
    return Norm(in_tsr, 0, n_procs)


class ReduceMean(Ops):
    def __init__(self, in_tsr, axis=None, keepdims=False, n_procs=None):
        if axis is None:
            axis = list(range(len(in_tsr)))
        elif not hasattr(axis, "__len__"):
            axis = list(axis)

        assert len(axis) <= len(in_tsr) and all(a < len(in_tsr) for a in axis)

        self.axis = axis
        self.keepdims = keepdims

        out_tsr = []
        axis = set(axis)
        for i, t in enumerate(in_tsr):
            if i not in axis:
                out_tsr.append(t)
            elif keepdims:
                out_tsr.append(1)
        if not out_tsr:
            out_tsr = (1,)
        super().__init__(in_tsr, in_tsr, Tensor(out_tsr), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        self.in_tsr_configs = self.dom_configs
        cols = list(set(range(len(self.GetInTensorConfigs(0)))) -
                set(self.axis))
        if not cols:
            if self.keepdims == True:
                self.out_tsr_configs = np.ones(self.dom_configs.shape)
            else:
                self.out_tsr_configs = np.ones((self.dom_configs.shape[0], 1))
        else:
            if self.keepdims == True:
                self.out_tsr_configs = np.ones(self.dom_configs.shape)
                self.out_tsr_configs[:, cols] = self.dom_configs[:, cols]
            else:
                self.out_tsr_configs = self.dom_configs[:, cols]
                if self.out_tsr_configs.ndim == 1:
                    self.out_tsr_configs = self.out_tsr_configs.reshape((-1,1))

        dom_per_proc = self.dom / self.dom_configs
        words = np.prod(dom_per_proc, axis=1)
        procs = np.prod(self.dom_configs[:, self.axis], axis=1)
        self.costs = GetAllReduceCost(words, procs)


class Softmax(Ops):
    def __init__(self, in_tsr, axis=1, n_procs=None):
        assert axis < len(in_tsr)
        self.axis = axis
        super().__init__(in_tsr, in_tsr, Tensor(in_tsr), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        # Softmax computation costs - Taking exponent and summation in forward
        # pass + cost of performing N multiplications per input in backward pass.
        exp_cost = 3.0 # Cost of computing a single exponent
        dom_per_proc = self.dom / self.dom_configs
        self.costs = (exp_cost + 1) * np.prod(dom_per_proc, axis=1)
        self.costs += np.prod(dom_per_proc, axis=1) * self.dom[self.axis]

        # Softmax communication costs - Adding partial sums: 1 word per input
        # per proc => batchsize / proc in forward pass.
        # Cost of gathering the rows in backward pass.
        elems = np.prod(np.delete(dom_per_proc, self.axis, 1), axis=1)
        self.costs = BytesToFlops(2.0 * elems)
        self.costs += GetAllReduceCost(elems * self.dom[self.axis],
                dom_per_proc[:, self.axis])


class SoftmaxCrossEntropy(Ops):
    def __init__(self, in_tsr, n_procs=None):
        assert len(in_tsr) == 2
        super().__init__(in_tsr, in_tsr, Tensor(in_tsr), n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        # Softmax computation costs - Taking exponent and summation in forward
        # pass + cost of performing one subtraction per input in backward pass.
        exp_cost = 3.0 # Cost of computing a single exponent
        dom_per_proc = self.dom / self.dom_configs
        self.costs = (exp_cost + 1) * np.prod(dom_per_proc, axis=1)
        self.costs += dom_per_proc[:, 0]

        # Cross-entropy computation costs - cost of averaging scalars over batch
        # dimension in forward pass.
        self.costs += dom_per_proc[:, 0]

        # Softmax communication costs - Adding partial sums: 1 word per input
        # per proc => batchsize / proc in forward pass.
        # Cross-entropy communication costs - Cost of broadcasting
        # the one label element per input along class dimension. Cost of
        # averaging partial sums: 1 word per proc along batch dim in forward
        # pass is ignored.
        # No communication in backward pass.
        comm_cost = BytesToFlops(2.0 * dom_per_proc[:, 0])
        np.add(self.costs, comm_cost, where=(self.dom_configs[:,1] > 1),
                out=self.costs)


class Embedding(Ops):
    def __init__(self, in_tsr, vocab_size, embedding_dim, n_procs=None):
        assert len(in_tsr) == 1

        self.embedding_dim = embedding_dim

        dom = (in_tsr[0], vocab_size)
        out_tsr = Tensor((in_tsr[0], embedding_dim))
        super().__init__(dom, in_tsr, out_tsr, n_procs)

    def ComputeCosts(self):
        super().ComputeCosts()

        self.in_tsr_configs = self.dom_configs[:, 0]
        self.out_tsr_configs = np.insert(self.dom_configs[:, 1].reshape(-1, 1),
                1, 1, axis=1)

        dom_per_proc = self.dom / self.dom_configs

        # Lookup in forward phase may involve a gather operation. We assume an
        # equal likelihood for each input row to be present in a processor.
        # p_b: no. of procs along batch dim; p_e: no. of procs along embed dim
        # Probability of having the row for an input in a processor 'p' is 1/p_e.
        # Each processor has b/p_b inputs. Hence, b/(p_b*p_e) input rows can be
        # expected to be present in the current processor, and (b/p_b)*(1-1/p_e)
        # elements have to be gathered from other processors.
        self.costs = BytesToFlops(dom_per_proc[:,0] * (1.0 - 1.0 /
            self.dom_configs[:,1]))

        # Gradient update costs
        weights_per_proc = dom_per_proc[:,1] * self.embedding_dim
        self.costs += weights_per_proc # Gradient addition
        self.costs += GetAllReduceCost(weights_per_proc, self.dom_configs[:, 0])

