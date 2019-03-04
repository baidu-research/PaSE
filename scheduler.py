import cProfile
import sys
import networkx as nx
import numpy as np
import pandas as pd

import itertools
from argparse import ArgumentParser

import graph


# Extends 'tbl' by adding configuration combinations of the vertices in
# 'vert_labels'
def ExtendTable(tbl, vert_labels, vert_ops):
    cols = set(tbl.columns.values)

    # If all the vertices are already present, just return the original table
    if set(vert_labels).issubset(cols):
        return tbl

    # Create missing columns and merge them with 'tbl'
    tbl_with_key = tbl.assign(key=0)
    for v in vert_labels:
        if v not in cols:
            v_df = pd.DataFrame(pd.Series(vert_ops[int(v)].dom_config_tuples,
                name=v))
            v_df = v_df.assign(key=0)
            if tbl_with_key.empty:
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
def ProcessGraph(G):
    g_tbl = pd.DataFrame()

    vert_ops = nx.get_node_attributes(G, 'op')
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')

    nodes = G.nodes()
    # TODO: Set the processing order to minimize the search space
    for v in nodes:
        vert_labels = [str(i) for i in itertools.chain(G.predecessors(v), G.successors(v))]

        # Extend the table with cartesian product of the neighbors
        g_tbl = ExtendTable(g_tbl, vert_labels, vert_ops)

        # Extend 'tbl' with column for 'v'
        tbl = ExtendTable(g_tbl, [str(v)], vert_ops)
        tbl = tbl[vert_labels + [str(v)]]

        # Get vertex costs for configs of 'v'
        v_idx = tbl[str(v)]
        cost_tbl = tbl.assign(costs = vert_costs[v].loc[v_idx].values)

        # Add edge cost of neighbors
        for n in G.predecessors(v):
            cost_tbl = AddEdgeCosts(n, v, edge_costs[(n, v)], cost_tbl)
        for n in G.successors(v):
            cost_tbl = AddEdgeCosts(v, n, edge_costs[(v, n)], cost_tbl)

        # Get the min cost for each neighbor sub-strategy
        cost_tbl.set_index(vert_labels, append=True, inplace=True)
        min_idx = cost_tbl['costs'].groupby(level=vert_labels,
                axis=0).idxmin(axis=0)
        min_idx = pd.MultiIndex.from_tuples(min_idx.values)
        assert(min_idx.levels[0].dtype == int)
        tbl = tbl.loc[min_idx.levels[0]]

        # Merge 'tbl' with 'g_tbl'
        merge_idx = vert_labels
        if str(v) in g_tbl.columns:
            merge_idx.append(str(v))
        g_tbl = g_tbl.merge(tbl, on=merge_idx, how='inner')

    return g_tbl


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--procs", type=int, required=False, default=8,
            help="No. of processors. (Default: 32)")
    parser.add_argument("-b", "--batch", type=int, required=False, default=128,
            help="Batch size. (Default: 128)")
    parser.add_argument("-m", "--model", type=int, required=False, default=128,
            help="Model size. (Default: 128)")
    parser.add_argument("-g", "--graph", type=str, required=False,
            choices=['test', 'alexnet', 'resnet101'], default='alexnet', 
            help="Neural net graph. (Default: 'alexnet')")
    parser.add_argument("--profile", dest="profile", action='store_true',
            help="Turn on/off profiling")
    parser.set_defaults(profile=False)
    args = vars(parser.parse_args())

    batch_size = args['batch']
    hidden_dim_size = args['model']
    n_procs = args['procs']

    # Profiling
    if args['profile']:
        pr = cProfile.Profile()

    # Create input graph
    G = graph.CreateGraph(args['graph'], batch_size, hidden_dim_size, n_procs)

    # Process the vertices
    if args['profile']:
        pr.enable()
    g_tbl = ProcessGraph(G)
    if args['profile']:
        pr.disable()
        pr.print_stats(sort='cumtime')

    cols = g_tbl.columns
    assert(len(cols) == G.number_of_nodes())
    print("Total strategies to check: " + str(g_tbl.shape[0]))
    sys.stdout.flush()

    # Iterate over all strategies and compute their cost
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')
    g_cost_tbl = g_tbl.assign(costs = 0)
    for v, v_c in G.nodes(data='costs'):
        g_cost_tbl = AddVertexCosts(v, v_c, g_cost_tbl)
    for u, v, e_c in G.edges(data='costs'):
        g_cost_tbl = AddEdgeCosts(u, v, e_c, g_cost_tbl)

    # Pick the strategy with min cost
    min_idx = g_cost_tbl['costs'].idxmin(axis=0)
    min_strategy = g_tbl.loc[min_idx]

    print("Strategy with minimum cost:")
    print("=====")
    print(min_strategy.to_string())
    print("=====")


if __name__ == "__main__":
    main()

