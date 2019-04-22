import networkx as nx
import numpy as np
import pandas as pd

import math
import itertools
from functools import reduce
import operator as op

import nn_ops


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


# Returns edge costs for different configs. Edge cost is computed as the
# difference b/w tensor volume needed per proc by the target vertex and the tensor
# volume held per proc by the source vertex.
def GetEdgeCosts(src_tsr, tgt_tsr, src_cfgs, tgt_cfgs, reshape):
    # Calculate the domains per processor
    src_tsr_per_proc = src_tsr / src_cfgs
    tgt_tsr_per_proc = tgt_tsr / tgt_cfgs

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
    costs = nn_ops.BytesToFlops(costs)

    return costs


def AddVertex(G, op):
    node_id = G.number_of_nodes()

    print("Node: " + str(node_id) + "; Configs: " +
            str(op.dom_configs.shape[0]))

    costs = op.GetCosts()
    costs = pd.Series(costs, index=op.dom_config_tuples, name='cost')

    G.add_node(node_id, op=op, costs=costs)
    return node_id


def AddEdge(G, src, tgt, src_tsr_idx=0, tgt_tsr_idx=0, reshape=None):
    assert(src in G)
    assert(tgt in G)

    node_ops = G.nodes(data='op')
    src_op = node_ops[src]
    tgt_op = node_ops[tgt]

    src_tsr = src_op.GetOutTensor(src_tsr_idx)
    tgt_tsr = tgt_op.GetInTensor(tgt_tsr_idx)
    src_cfgs = src_op.GetOutTensorConfigs(src_tsr_idx)
    tgt_cfgs = tgt_op.GetInTensorConfigs(tgt_tsr_idx)

    costs = GetEdgeCosts(src_tsr, tgt_tsr, src_cfgs, tgt_cfgs, reshape)
    idx = pd.MultiIndex.from_product([src_op.dom_config_tuples,
        tgt_op.dom_config_tuples], names=[str(src), str(tgt)])
    costs = pd.Series(costs, index=idx, name='cost')

    G.add_edge(src, tgt, costs=costs)


def AlexNet(G, b):
    img = nn_ops.Tensor((b, 3, 227, 227))

    # Conv1 + relu + maxpool
    conv1 = nn_ops.Conv(img, (96, 3, 11, 11), stride=4, pw_op_cnt=1)
    node_id1_0 = AddVertex(G, conv1)
    pool1 = nn_ops.Pooling(conv1.GetOutTensor(0), (3, 3), stride=2)
    node_id1 = AddVertex(G, pool1)
    AddEdge(G, node_id1_0, node_id1)

    # Conv2 + relu + maxpool
    conv2 = nn_ops.Conv(pool1.GetOutTensor(0), (256, 96, 5, 5), pad=2,
            pw_op_cnt=1)
    node_id2_0 = AddVertex(G, conv2)
    AddEdge(G, node_id1, node_id2_0)
    pool2 = nn_ops.Pooling(conv2.GetOutTensor(0), (3, 3), stride=2)
    node_id2 = AddVertex(G, pool2)
    AddEdge(G, node_id2_0, node_id2)

    # Conv3 + relu
    conv3 = nn_ops.Conv(pool2.GetOutTensor(0), (384, 256, 3, 3), stride=1,
            pad=1, pw_op_cnt=1)
    node_id3 = AddVertex(G, conv3)
    AddEdge(G, node_id2, node_id3)

    # Conv4 + relu
    conv4 = nn_ops.Conv(conv3.GetOutTensor(0), (384, 384, 3, 3), stride=1,
            pad=1, pw_op_cnt=1)
    node_id4 = AddVertex(G, conv4)
    AddEdge(G, node_id3, node_id4)

    # Conv5 + relu + maxpool
    conv5 = nn_ops.Conv(conv4.GetOutTensor(0), (256, 384, 3, 3), stride=1,
            pad=1, pw_op_cnt=1)
    node_id5_0 = AddVertex(G, conv5)
    AddEdge(G, node_id4, node_id5_0)
    pool5 = nn_ops.Pooling(conv5.GetOutTensor(0), (3, 3), stride=2)
    node_id5 = AddVertex(G, pool5)
    AddEdge(G, node_id5_0, node_id5)

    # FC6 + relu
    tsr = pool5.GetOutTensor(0)
    tsr = (tsr[0], tsr[1] * tsr[2] * tsr[3])
    fc6 = nn_ops.FC(tsr, 4096, pw_op_cnt=1)
    node_id6 = AddVertex(G, fc6)
    AddEdge(G, node_id5, node_id6, reshape=[(0,), (1,2,3)])

    # FC7 + relu
    fc7 = nn_ops.FC(fc6.GetOutTensor(0), 4096, pw_op_cnt=1)
    node_id7 = AddVertex(G, fc7)
    AddEdge(G, node_id6, node_id7)

    # FC8
    fc8 = nn_ops.FC(fc7.GetOutTensor(0), 1024)
    node_id8 = AddVertex(G, fc8)
    AddEdge(G, node_id7, node_id8)

    # Softmax + cross-entropy loss
    loss = nn_ops.SoftmaxCrossEntropy(fc8.GetOutTensor(0))
    node_id9 = AddVertex(G, loss)
    AddEdge(G, node_id8, node_id9)

    return G


# Creates the graph for the model
def CreateGraph(graph_type, batch_size, hidden_dim_size, n_procs):
    G = nx.DiGraph()
    nn_ops.Ops.default_procs = n_procs

    if graph_type == 'alexnet':
        G = AlexNet(G, batch_size)
    elif graph_type == 'resnet101':
        G = ResNet101(G, batch_size, n_procs).Graph()
    elif graph_type == 'inception3':
        G = Inception3(G, batch_size, n_procs).Graph()
    elif graph_type == 'seq2seq':
        G = Seq2seq(G, batch_size, n_procs)
    else:
        assert(False)

    return G

