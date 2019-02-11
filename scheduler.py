import networkx as nx
import numpy as np
import pandas as pd

import operator
from functools import partial
from sortedcontainers import SortedList
import itertools
import copy as cp

import cost_np as cst
import config as cfg
import utils


def AssignCostsToNodes(G, n_procs):
    for v, attr in G.nodes(data=True):
        dom = attr['dom']

        config_tuples = cfg.GetNodeConfigs(dom, n_procs)
        configs = np.array(config_tuples)
        costs = cst.GetVertexCosts(np.array(dom), configs)
        idx = pd.Index(configs, dtype=tuple, name=str(v))

        attr['config_tuples'] = config_tuples
        attr['configs'] = configs
        attr['costs'] = pd.Series(costs, index=idx, name='cost')


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


def CreateGraph(batch_size, hidden_dim_size):
    G = nx.DiGraph()
    G.add_node(1, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(2, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(3, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    return G


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


def GetVertexIndices(v, g_tbl, vert_costs):
    if str(v) in g_tbl.columns:
        in_tbl = True
        v_idx = pd.Index(g_tbl.loc[:, str(v)])
    else:
        in_tbl = False
        v_idx = vert_costs[v].index

    return in_tbl, v_idx


def AddEdgeCosts(src, tgt, edge_costs, tbl):
    idx = pd.Index(tbl[[str(src), str(tgt)]])
    edge_cost = edge_costs[(src, tgt)].loc[idx]
    tbl['costs'] += edge_cost.values

    return tbl


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
        tbl = AddEdgeCosts(n, v, edge_costs, tbl)
    for n in G.successors(v):
        tbl = AddEdgeCosts(v, n, edge_costs, tbl)

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
    batch_size = 32
    hidden_dim_size = 32
    n_procs = 4

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

    # Iterate over all strategies and compute their cost
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')
    g_tbl = g_tbl.assign(costs = 0)
    for v, v_c in G.nodes(data='costs'):
        idx = pd.Index(g_tbl[str(v)])
        vert_cost = v_c.loc[idx]
        g_tbl['costs'] += vert_cost.values


    # Pick the strategy with min cost
    min_idx = g_tbl['costs'].idxmin()
    min_strategy = g_tbl.drop('costs', 1).loc[min_idx]

    print("Strategy with minimum cost:")
    print(min_strategy)


if __name__ == "__main__":
    main()

