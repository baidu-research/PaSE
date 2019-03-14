import cProfile
import sys
import time
import networkx as nx
import numpy as np
import pandas as pd

import itertools
from argparse import ArgumentParser

import graph


def MergeTwoTables(tbl1, tbl2):
    common_keys = list(set(tbl1.columns).intersection(tbl2.columns))

    if not common_keys:
        if 'key' not in tbl1.columns:
            tbl1 = tbl1.assign(key=0)
        if 'key' not in tbl2.columns:
            tbl2 = tbl2.assign(key=0)
        return tbl1.merge(tbl2, on='key').drop('key', 1)
    else:
        return tbl1.merge(tbl2, on=common_keys)


def MergeTables(tbls):
    n_tbls = len(tbls)

    if n_tbls == 0:
        return None

    if n_tbls == 1:
        return tbls[0]

    m_tbl = MergeTwoTables(tbls[0], tbls[1])

    for tbl in tbls[2:]:
        m_tbl = MergeTwoTables(m_tbl, tbl)

    return m_tbl


# Adds vertex costs of 'v' to the table
def AddVertexCosts(v, vert_costs, tbl):
    tbl = tbl.merge(vert_costs, left_on=[str(v)], right_index=True, how='left')
    try:
        tbl['costs'] += tbl['cost']
        tbl.drop('cost', 1, inplace=True)
    except KeyError:
        tbl.rename(columns={'cost': 'costs'}, inplace=True)

    return tbl


# Adds edge costs of '(src, tgt)' to the table
def AddEdgeCosts(src, tgt, edge_costs, tbl):
    tbl = tbl.merge(edge_costs, on=[str(src), str(tgt)], how='left')
    tbl['costs'] += tbl['cost']
    tbl.drop('cost', 1, inplace=True)

    return tbl


# Sorts nodes in the ascending order of no. of unprocessed neighbors.
def SortNodes(G):
    n_nodes = G.number_of_nodes()

    # Create a table with node id, unprocessed neighbor count, and processed
    # neighbor count
    node_tbl = [(v, cnt, 0) for v, cnt in G.degree]
    node_tbl = np.array(node_tbl)
    assert(node_tbl.shape[0] == n_nodes)

    # Sort the array by degree
    node_tbl.view('i8,i8,i8').sort(order=['f1'], axis=0)
    node_idx = np.empty(n_nodes, dtype=int)
    for i, v in enumerate(node_tbl[:,0]):
        node_idx[v] = i

    for i in range(n_nodes):
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
        

def ReduceTable(tbl, grouping_cols, minimization_col):
    if grouping_cols:
        min_idx = tbl.groupby(grouping_cols,
                axis=0)[minimization_col].idxmin(axis=0)
    else:
        min_idx = tbl[minimization_col].idxmin(axis=0)

    if len(min_idx.shape) == 0:
        min_idx = [min_idx]

    return tbl.loc[min_idx]


class Processor:
    def __init__(self, G):
        self.n_nodes = G.number_of_nodes()

        self.G = G
        self.v_to_tbl_map = self.n_nodes * [None]
        self.processed_nodes = set()
        self.processed_node_labels = set()

        self.vert_ops = nx.get_node_attributes(G, 'op')
        self.vert_costs = nx.get_node_attributes(self.G, 'costs')
        self.edge_costs = nx.get_edge_attributes(self.G, 'costs')

    # Generates table for vertex 'v'
    def GenerateTable(self, v, p_neigh, up_neigh):
        cfg_to_df = lambda x : pd.DataFrame().assign(**{str(x) :
            self.vert_ops[x].dom_config_tuples})

        # Merge tables of processed neighbors and add configurations of 'v' to 'tbl'
        if p_neigh:
            tbl = MergeTables([self.v_to_tbl_map[n] for n in p_neigh])
            assert(tbl.shape[0] > 0)

        try:
            if str(v) not in tbl.columns:
                tbl = MergeTwoTables(tbl, cfg_to_df(v))
        except NameError:
            tbl = cfg_to_df(v)

        # Add all combinations of configurations of unprocessed neighbors
        cols = set(tbl.columns)
        for n in up_neigh:
            if str(n) not in cols:
                tbl = MergeTwoTables(tbl, cfg_to_df(n))

        assert(tbl.shape[0] > 0)
        return tbl

    # Compute costs for sub-strategies in 'tbl'
    def ComputeCosts(self, tbl, v, p_preds, p_scsrs, up_preds, up_scsrs):
        tbl = AddVertexCosts(v, self.vert_costs[v], tbl)

        for n in up_preds:
            tbl = AddEdgeCosts(n, v, self.edge_costs[(n, v)], tbl)
        for n in up_scsrs:
            tbl = AddEdgeCosts(v, n, self.edge_costs[(v, n)], tbl)

        tbl.rename(columns={'costs' : 'costs_' + str(v)}, inplace=True)

        return tbl

    def ProcessVertex(self, v):
        preds = set(self.G.predecessors(v))
        scsrs = set(self.G.successors(v))

        p_preds = self.processed_nodes.intersection(preds)
        p_scsrs = self.processed_nodes.intersection(scsrs)
        p_neigh = p_preds.union(p_scsrs)

        up_preds = preds - p_preds
        up_scsrs = scsrs - p_scsrs
        up_neigh = up_preds.union(up_scsrs)

        assert(v not in self.processed_nodes)

        # Add 'v' to processed node set
        self.processed_nodes.add(v)
        self.processed_node_labels.add(str(v))

        # Create the table for 'v' by merging neighbor tables, and compute costs
        # for different sub-strategies
        tbl = self.GenerateTable(v, p_neigh, up_neigh)
        tbl = self.ComputeCosts(tbl, v, p_preds, p_scsrs, up_preds, up_scsrs)

        print("Processing vertex " + str(v) + "; Table size: " +
                str(tbl.shape[0]))

        # Add individual sub-strategy costs to get complete sub-strategy costs
        # for 'tbl'
        cost_col_name = 'costs_' + str(v)
        for c in tbl.columns:
            if c.startswith('costs_') and c != cost_col_name:
                tbl[cost_col_name] += tbl[c]
                tbl.drop(c, 1, inplace=True)

        cols = set(tbl.columns) - self.processed_node_labels
        cols.remove(cost_col_name)
        col_names = [str(c) for c in cols]
        tbl = ReduceTable(tbl, col_names, cost_col_name)

        # Update the vertex to table map
        for c in tbl.columns:
            try:
                self.v_to_tbl_map[int(c)] = tbl
            except ValueError:
                assert(c.startswith('costs_'))

        print("Processed vertex " + str(v) + "; Table size: " +
                str(tbl.shape[0]) + "\n")
        return tbl

    def ProcessGraph(self, vert_order):
        for v in vert_order:
            tbl = self.ProcessVertex(v)

        assert(len(tbl.columns) == self.n_nodes + 1)
        return tbl

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--procs", type=int, required=False, default=8,
            help="No. of processors. (Default: 32)")
    parser.add_argument("-b", "--batch", type=int, required=False, default=128,
            help="Batch size. (Default: 128)")
    parser.add_argument("-m", "--model", type=int, required=False, default=128,
            help="Model size. (Default: 128)")
    parser.add_argument("-g", "--graph", type=str, required=False,
            choices=['test', 'alexnet', 'resnet101', 'inception3'],
            default='alexnet', help="Neural net graph. (Default: 'alexnet')")
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
    g_tbl = Processor(G).ProcessGraph(SortNodes(G))
    print("")
    if args['profile']:
        pr.disable()
        pr.print_stats(sort='cumtime')

    # Convert 'g_tbl' into Series from DataFrame
    assert(g_tbl.shape[0] == 1)
    g_tbl = g_tbl.iloc[0]

    print("Strategy with minimum cost:")
    print("=====")
    print(g_tbl.to_string())
    print("=====")


if __name__ == "__main__":
    main()

