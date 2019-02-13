import networkx as nx
import numpy as np
import pandas as pd

import operator as op
from functools import reduce
from sortedcontainers import SortedList
import itertools
import math
from argparse import ArgumentParser

import cost as cst


# Generates list of configurations for a vertex
def GetNodeConfigs(node_dom, n_procs):
    dim = len(node_dom)
    log_n_procs = int(math.log2(n_procs))
    procs = [1 << i for i in range(log_n_procs + 1)]

    configs = [c for c in itertools.product(procs, repeat=dim) if reduce(op.mul,
        c, 1) <= n_procs]
    return configs


# Computes vertex costs for different configurations of vertices, and assigns
# them to the vertices
def AssignCostsToNodes(G, n_procs):
    for v, attr in G.nodes(data=True):
        dom = attr['dom']

        config_tuples = GetNodeConfigs(dom, n_procs)
        configs = np.array(config_tuples)
        costs = cst.GetVertexCosts(np.array(dom), configs)
        idx = pd.Index(configs, dtype=tuple, name=str(v))

        attr['config_tuples'] = config_tuples
        attr['configs'] = configs
        attr['costs'] = pd.Series(costs, index=idx, name='cost')


# Computes edge costs for different configs, and assigns them to edges
def AssignCostsToEdges(G, n_procs):
    nodes = G.nodes(data=True)

    for src, tgt, edge_attr in G.edges(data=True):
        src_attr = nodes[src]
        tgt_attr = nodes[tgt]

        src_dom = np.array(src_attr['dom'])
        tgt_dom = np.array(tgt_attr['dom'])
        src_config_tuples = src_attr['config_tuples']
        tgt_config_tuples = tgt_attr['config_tuples']
        src_configs = src_attr['configs']
        tgt_configs = tgt_attr['configs']
        src_costs = src_attr['costs']
        tgt_costs = tgt_attr['costs']

        costs = cst.GetEdgeCosts(src_dom, tgt_dom, src_configs, tgt_configs)
        idx = pd.MultiIndex.from_product([src_config_tuples, tgt_config_tuples],
                names=[str(src), str(tgt)])
        edge_attr['costs'] = pd.Series(costs, index=idx, name='cost')


# Creates the graph for the model
def CreateGraph(batch_size, hidden_dim_size):
    G = nx.DiGraph()
    G.add_node(1, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(2, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(3, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    return G


# Extends 'tbl' by adding configuration combinations of the vertices in
# 'vert_labels'
def ExtendTable(tbl, vert_labels, vert_cfgs):
    cols = tbl.columns

    # If all the vertices are already present, just return the original table
    if set(vert_labels).issubset(cols):
        return tbl

    tbl_with_key = tbl.assign(key=0)
    for v in vert_labels:
        if v not in cols:
            v_df = pd.DataFrame(pd.Series(vert_cfgs[int(v)], name=v))
            v_df = v_df.assign(key=0)
            if tbl.empty:
                tbl_with_key = v_df
            else:
                tbl_with_key = tbl_with_key.merge(v_df, on ='key')

    return tbl_with_key.drop('key', 1)


# Adds vertex costs of 'v' to the table
def AddVertexCosts(v, vert_costs, tbl):
    idx = pd.Index(tbl[str(v)])
    vert_cost = vert_costs.loc[idx]
    tbl['costs'] += vert_cost.values

    return tbl


# Adds edge costs of '(src, tgt)' to the table
def AddEdgeCosts(src, tgt, edge_costs, tbl):
    idx = pd.Index(tbl[[str(src), str(tgt)]])
    edge_cost = edge_costs.loc[idx]
    tbl['costs'] += edge_cost.values

    return tbl


# Processes vertex 'v'
def ProcessVertex(G, v):
    g_tbl = G.graph['tbl']

    vert_cfgs = nx.get_node_attributes(G, 'config_tuples')
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')

    vert_labels = [str(i) for i in itertools.chain(G.predecessors(v), G.successors(v))]

    # Extend the table with cartesian product of the neighbors
    g_tbl = ExtendTable(g_tbl, vert_labels, vert_cfgs)

    # Extend 'tbl' with column for 'v'
    tbl = ExtendTable(g_tbl, [str(v)], vert_cfgs)
    tbl = tbl[vert_labels + [str(v)]]

    # Get vertex costs for configs of 'v'
    v_idx = tbl[str(v)]
    tbl = tbl.assign(costs = vert_costs[v].loc[v_idx].values)

    # Add edge cost of neighbors
    for n in G.predecessors(v):
        tbl = AddEdgeCosts(n, v, edge_costs[(n, v)], tbl)
    for n in G.successors(v):
        tbl = AddEdgeCosts(v, n, edge_costs[(v, n)], tbl)

    # Get the min cost for each neighbor sub-strategy
    tbl.set_index(vert_labels + [str(v)], inplace=True)
    idx_names = tbl.index.names
    assert(len(tbl.columns) == 1)
    min_idx = tbl.groupby(level=vert_labels)['costs'].idxmin()
    min_idx = pd.MultiIndex.from_tuples(min_idx.values)
    tbl = tbl.loc[min_idx]
    tbl.index.names = idx_names
    tbl.reset_index(str(v), inplace=True)
    tbl.drop('costs', 1, inplace=True)

    # Merge 'tbl' with 'g_tbl'
    merge_idx = vert_labels
    if str(v) in g_tbl.columns:
        merge_idx.append(str(v))
    g_tbl = g_tbl.merge(tbl, on=merge_idx, how='inner')
    G.graph['tbl'] = g_tbl


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--procs", type=int, required=False, default=8,
            help="No. of processors. (Default: 32)")
    parser.add_argument("-b", "--batch", type=int, required=False, default=128,
            help="Batch size. (Default: 128)")
    parser.add_argument("-m", "--model", type=int, required=False, default=128,
            help="Model size. (Default: 128)")
    args = vars(parser.parse_args())

    batch_size = args['batch']
    hidden_dim_size = args['model']
    n_procs = args['procs']

    # Create input graph
    G = CreateGraph(batch_size, hidden_dim_size)
    G.graph['tbl'] = pd.DataFrame()

    # Assign config list to nodes in 'G', and their costs
    AssignCostsToNodes(G, n_procs)

    # Assign configs and costs for each edge
    AssignCostsToEdges(G, n_procs)

    # Process the vertices
    for v in G.nodes():
        ProcessVertex(G, v)

    g_tbl = G.graph['tbl']
    cols = g_tbl.columns
    assert(len(cols) == G.number_of_nodes())
    print("Total strategies to check: " + str(g_tbl.shape[0]))

    # Iterate over all strategies and compute their cost
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')
    g_tbl = g_tbl.assign(costs = 0)
    for v, v_c in G.nodes(data='costs'):
        g_tbl = AddVertexCosts(v, v_c, g_tbl)
    for u, v, e_c in G.edges(data='costs'):
        g_tbl = AddEdgeCosts(u, v, e_c, g_tbl)

    # Pick the strategy with min cost
    min_idx = g_tbl['costs'].idxmin()
    min_strategy = g_tbl.drop('costs', 1).loc[min_idx]

    print("Strategy with minimum cost:")
    print("=====")
    print(min_strategy.to_string())
    print("=====")


if __name__ == "__main__":
    main()

