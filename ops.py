import numpy as np
import pandas as pd

import math
import itertools
from functools import reduce


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

    configs = [c for c in itertools.product(*proc_set) if reduce(op.mul, c, 1)
        <= n_procs]
    return configs


def ComputeGemmCosts(dom, dom_configs, pw_op_cnt):
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


def GetConvolutedSize(h, w, r, s, stride, pad):
    stride_r, stride_s = MakePair(stride)
    pad_r, pad_s = MakePair(pad)

    h_o = int((h - r + 2*pad_r) / stride_r) + 1
    w_o = int((w - s + 2*pad_s) / stride_s) + 1

    return h_o, w_o


# Parent operator class
class Ops():
    def __init__(self, dom, in_tsrs, out_tsrs, n_procs):
        assert all(hasattr(i, '__len__') for i in self.in_tsrs)
        assert all(hasattr(i, '__len__') for i in self.out_tsrs)

        self.dom = tuple(dom)
        self.in_tsrs = tuple(in_tsrs)
        self.out_tsrs = tuple(out_tsrs)
        self.n_procs = n_procs

    def ComputeCosts(self):
        self.dom_config_tuples = None
        self.dom_configs = None
        self.in_tsr_configs = None
        self.out_tsr_configs = None
        self.costs = 0

    # Returns vertex costs for different configs
    def GetCosts(self):
        try:
            return self.costs
        except AttributeError:
            self.ComputeCosts()
            return self.costs


# GEMM
class Gemm(Ops):
    def __init__(self, dom, n_procs, pw_op_cnt=0):
        assert len(dom) == 3

        self.pw_op_cnt = pw_op_cnt

        m_idx, n_idx, k_idx = range(3)

        # Domain and input/output tensors
        in_tsr = (dom[m_idx], dom[k_idx])
        out_tsr = (dom[m_idx], dom[n_idx])
        super().__init__(dom, (in_tsr,), (out_tsr,), n_procs)

    def ComputeCosts(self):
        m_idx, n_idx, k_idx = range(3)

        # Configurations
        self.dom_config_tuples = GetConfigs(self.dom, self.n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs[:, (m_idx, k_idx)]
        self.out_tsr_configs = self.dom_configs[:, m_idx:n_idx+1]

        # Compute the costs for configs
        self.costs = ComputeGemmCosts(self.dom, self.dom_configs, self.pw_op_cnt)


# Convolution
class Conv(Ops):
    def __init__(self, img, fltr, n_procs, stride=1, pad=0, pw_op_cnt=0):
        assert len(img) == 4
        assert len(fltr) == 4
        assert img[1] == fltr[1]

        self.pw_op_cnt = pw_op_cnt

        b, c, h, w = img
        n, _, r, s = fltr
        h_o, w_o = GetConvolutedSize(h, w, r, s, stride, pad)

        # Domain
        dom = (b, c, h_o, w_o, r, s, n)
        out_tsr = (b, n, h_o, w_o)
        super().__init__(dom, (img,), (out_tsr,), n_procs)

    def ConvertToGemmDom(self):
        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)
        b, c, h_o, w_o, r, s, n = self.dom

        gemm_dom = (b * h_o * w_o, n, c * r * s)
        gemm_m = np.prod(self.dom_configs[:, (b_idx, h_idx, w_idx)],
                axis=1).reshape(-1, 1)
        gemm_n = self.dom_configs[:, n_idx].reshape(-1, 1)
        gemm_k = np.prod(self.dom_configs[:, (c_idx, r_idx, s_idx)],
                axis=1).reshape(-1, 1)
        gemm_configs = np.concatenate((gemm_m, gemm_n, gemm_k), axis=1)

        return gemm_dom, gemm_configs

    # Fuse maxpool
    def Fuse(self, maxpool, after=True):
        assert isinstance(maxpool, MaxPool)
        # Make sure we haven't computed configs yet
        assert not hasattr(self, 'dom_configs')

        try:
            maxpool_dom_configs = maxpool.dom_configs
        except AttributeError:
            maxpool.ComputeCosts()
            maxpool_dom_configs = maxpool.dom_configs

        # Compute original configs
        dom_config_tuples = GetConfigs(self.dom, self.n_procs)
        dom_configs = np.array(self.dom_config_tuples)
        dom_configs = dom_configs[np.all(np.isin(dom_configs[:,:4],
            maxpool_dom_configs), axis=1), :]

        # Get configs that intersect with maxpool's configs
        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)
        self.dom_configs = dom_configs
        self.dom_config_tuples = [tuple(e) for e in dom_configs]
        self.in_tsr_configs = self.dom_configs[:, b_idx:w_idx+1]
        self.out_tsr_configs = self.dom_configs[:, (b_idx, n_idx, h_idx, w_idx)]

        # Compute costs
        gemm_dom, gemm_configs = self.ConvertToGemmDom()
        self.costs = ComputeGemmCosts(gemm_dom, gemm_configs, pw_op_cnt)

        # Add maxpool's costs
        assert self.costs.shape == maxpool.costs.shape
        self.costs += maxpool.costs

    def ComputeCosts(self):
        b_idx, c_idx, h_idx, w_idx, r_idx, s_idx, n_idx = range(7)

        # Configurations
        self.dom_config_tuples = GetConfigs(self.dom, self.n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs[:, b_idx:w_idx+1]
        self.out_tsr_configs = self.dom_configs[:, (b_idx, n_idx, h_idx, w_idx)]

        # Represent convolution as GEMM computation, and compute the cost for
        # GEMM op
        gemm_dom, gemm_configs = self.ConvertToGemmDom()
        self.costs = ComputeGemmCosts(gemm_dom, gemm_configs, pw_op_cnt)


# Maxpool
class MaxPool(Ops):
    def __init__(self, img, fltr, n_procs, stride=1, pad=0):
        assert len(img) == 4
        assert len(fltr) == 2

        b, c, h, w = img
        r, s = fltr
        h_o, w_o = GetConvolutedSize(h, w, r, s, stride, pad)

        dom = (b, c, h_0, w_o)
        super().__init__(dom, (img,), (dom,), n_procs)

    def ComputeCosts(self):
        self.dom_config_tuples = GetConfigs(self.dom, self.n_procs)
        self.dom_configs = np.array(self.dom_config_tuples)
        self.in_tsr_configs = self.dom_configs
        self.out_tsr_configs = self.dom_configs

        dom_per_proc = self.dom / self.dom_configs
        self.costs = 3.0 * np.prod(dom_per_proc, axis=1)


