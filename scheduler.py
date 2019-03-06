import cProfile
import sys
import time
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
                tbl_with_key = tbl_with_key.merge(v_df, on='key')

    return tbl_with_key.drop('key', 1)


# Adds vertex costs of 'v' to the table
def AddVertexCosts(v, vert_costs, tbl):
    tbl = tbl.merge(vert_costs, left_on=[str(v)], right_index=True, how='left')
    if 'costs' in tbl:
        tbl['costs'] += tbl['cost']
        tbl.drop('cost', 1, inplace=True)
    else:
        tbl.rename(columns={'cost': 'costs'}, inplace=True)

    assert(tbl['costs'].isnull().values.any() == False)
    return tbl


# Adds edge costs of '(src, tgt)' to the table
def AddEdgeCosts(src, tgt, edge_costs, tbl):
    tbl = tbl.merge(edge_costs, on=[str(src), str(tgt)], how='left')
    tbl['costs'] += tbl['cost']
    tbl.drop('cost', 1, inplace=True)

    assert(tbl['costs'].isnull().values.any() == False)
    return tbl


# Sorts nodes in the ascending order of no. of unprocessed neighbors.
def SortNodes(G):
    n_nodes = G.number_of_nodes()

    # Create a table with node id, unprocessed neighbor count, and processed
    # neighbor count
    node_tbl = [(v, cnt, 0) for v, cnt in G.degree]
    node_tbl = np.array(node_tbl)

    # Sort the array by degree
    node_tbl.view('i8,i8,i8').sort(order=['f1'], axis=0)
    node_idx = np.empty(n_nodes, dtype=int)
    for i, v in enumerate(node_tbl[:,0]):
        node_idx[v] = i

    sz = node_tbl.shape[0]
    for i in range(sz):
        r = node_tbl[i]
        v = r[0]
        r[1] = -1 # Reset deg so that the sorting below doesn't move neighbors
                  # above this row

        yield v

        for neigh in itertools.chain(G.predecessors(v), G.successors(v)):
            neigh_idx = node_idx[neigh]

            # Update the neighbor only if it hasn't been processed already
            if neigh_idx > i:
                # Decrement the unprocessed neighbor count of neighbor, and
                # increment the processed neighbor count
                node_tbl[neigh_idx, 1] -= 1
                node_tbl[neigh_idx, 2] += 1
                up_cnt = node_tbl[neigh_idx, 1] # Unprocessed neigh cnt
                p_cnt = node_tbl[neigh_idx, 2] # Processed neigh cnt

                # Find the new index in the sorted array
                new_idx = neigh_idx
                while True:
                    curr_up_cnt, curr_p_cnt = node_tbl[new_idx-1, 1:3]
                    if (curr_up_cnt > up_cnt) or ((curr_up_cnt == up_cnt) and
                            (curr_p_cnt < p_cnt)):
                        new_idx -= 1
                    else:
                        break

                # Swap the row to its new position
                if new_idx != neigh_idx:
                    assert(new_idx > i)
                    node_idx[neigh] = new_idx
                    node_idx[node_tbl[new_idx, 0]] = neigh_idx
                    node_tbl[[neigh_idx, new_idx]] = node_tbl[[new_idx,
                        neigh_idx]]
        

# Processes vertex 'v'
def ProcessGraph(G):
    g_tbl = pd.DataFrame()

    vert_ops = nx.get_node_attributes(G, 'op')
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')

    sorted_nodes = SortNodes(G)
    for v in sorted_nodes:
        vert_labels = [str(i) for i in itertools.chain(G.predecessors(v), G.successors(v))]

        # Extend the table with cartesian product of the neighbors
        g_tbl = ExtendTable(g_tbl, vert_labels, vert_ops)

        # Extend 'tbl' with column for 'v'
        tbl = ExtendTable(g_tbl, [str(v)], vert_ops)
        tbl = tbl[vert_labels + [str(v)]]

        # Get vertex costs for configs of 'v'
        cost_tbl = AddVertexCosts(v, vert_costs[v], tbl)

        # Add edge cost of neighbors
        #start_time = time.time()
        for n in G.predecessors(v):
            cost_tbl = AddEdgeCosts(n, v, edge_costs[(n, v)], cost_tbl)
        for n in G.successors(v):
            cost_tbl = AddEdgeCosts(v, n, edge_costs[(v, n)], cost_tbl)
        #end_time = time.time()
        #print("Time: " + str(end_time - start_time))

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

        print("Processed vertex " + str(v) + "; Current table size: " +
                str(g_tbl.shape[0]))

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
            help="Turn on/off profiling.")
    parser.add_argument("-d", "--dump-graph", dest="dump_graph",
            action='store_true', help="Dump the graph in dot format to the file "
            "graph.dot in the working directory.")
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
    print("")

    if args['dump_graph']:
        try:
            import pydot
            from networkx.drawing.nx_pydot import write_dot
            write_dot(G, 'graph.dot')
            print("Graph written to graph.dot.\n")
        except ImportError:
            print("pydot package not found.")
            raise

    # Process the vertices
    if args['profile']:
        pr.enable()
    g_tbl = ProcessGraph(G)
    print("")
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

