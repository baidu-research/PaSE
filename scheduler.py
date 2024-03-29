import cProfile
import sys
import time
import datetime as dt
import networkx as nx
import numpy as np
import pandas as pd

import itertools
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import graph, nn_ops


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


def MergeTables(tbl1, tbl2):
    common_keys = list(set(tbl1.columns).intersection(tbl2.columns))

    if __debug__:
        if common_keys:
            tbl1_common = tbl1[common_keys].copy().drop_duplicates(
                            ).sort_values(by=common_keys,
                                    axis=0).reset_index(drop=True)
            tbl2_common = tbl2[common_keys].copy().drop_duplicates(
                            ).sort_values(by=common_keys,
                                    axis=0).reset_index(drop=True)
            assert (tbl1_common == tbl2_common).all(axis=None)

    if not common_keys:
        if 'key' not in tbl1.columns:
            tbl1 = tbl1.assign(key=0)
        if 'key' not in tbl2.columns:
            tbl2 = tbl2.assign(key=0)
        return tbl1.merge(tbl2, on='key').drop('key', 1)
    else:
        return tbl1.merge(tbl2, on=common_keys)

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

    def SortNodes(self):
        # Create a table with node_id and no. of unprocessed
        # ancestors/descendents
        node_tbl = np.array([(v, cnt) for v, cnt in self.G.degree])

        # Maintain a dictionary of {node_id: set(node_ids)} s.t. node_id depends
        # on unprocessed nodes in node_ids.
        node_dict = dict()
        for v in self.G.nodes():
            neighs = set(itertools.chain(self.G.predecessors(v),
                self.G.successors(v)))
            node_dict[v] = neighs

        for i in range(self.n_nodes):
            # Return the node_id with minimum count in node_tbl
            min_idx = node_tbl[:,1].argmin()
            node_id, cnt = node_tbl[min_idx]
            assert(cnt < self.n_nodes)
            yield node_id

            # Invalidate the count for node_id in node_tbl
            node_tbl[min_idx, 1] = self.n_nodes

            # Update node_tbl and node_dict for nodes that are affected by
            # node_id
            node_set = node_dict[node_id]
            for v in node_set:
                assert(node_tbl[v, 1] < self.n_nodes) # v has to be unprocessed
                s = node_dict[v].union(node_set)
                s = s - {node_id, v}
                node_dict[v] = s
                node_tbl[v, 1] = len(s)

    def SortNodesNaive(self):
        rset = set()
        uset = set(self.G.nodes())

        node_dict = dict()
        for v in self.G.nodes():
            neighs = set(itertools.chain(self.G.predecessors(v),
                self.G.successors(v)))
            node_dict[v] = neighs

        for i in range(self.n_nodes):
            node_id = None
            rset_size = None
            for v in uset:
                l = len(rset | node_dict[v] | {v})
                try:
                    if l < rset_size:
                        node_id = v
                        rset_size = l
                except TypeError:
                    node_id = v
                    rset_size = l

            yield node_id

            rset |= node_dict[node_id]
            rset.discard(node_id)
            uset.discard(node_id)
            for v in node_dict[node_id]:
                node_dict[v].discard(node_id)

            assert rset <= uset

    # Convert configuration series to dataframe
    def CfgToDf(self, v):
        return pd.DataFrame().assign(**{str(v) :
            self.vert_ops[v].dom_config_tuples})

    # Generates table for vertex 'v'
    def GenerateTable(self, v, p_neigh, up_neigh):
        # Merge tables of processed neighbors and add configurations of 'v' to 'tbl'
        if p_neigh:
            it = iter(p_neigh)
            tbl = self.v_to_tbl_map[next(it)]
            for n in it:
                tbl = MergeTables(tbl, self.v_to_tbl_map[n])
            assert(tbl.shape[0] > 0)
            assert str(v) in tbl.columns
        else:
            tbl = self.CfgToDf(v)

        # Add all combinations of configurations of unprocessed neighbors
        cols = set(tbl.columns)
        for n in up_neigh:
            if str(n) not in cols:
                tbl = MergeTables(tbl, self.CfgToDf(n))

        assert(tbl.shape[0] > 0)
        return tbl

    # Compute costs for sub-strategies in 'tbl'
    def ComputeCosts(self, tbl, v, up_preds, up_scsrs):
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
        tbl = self.ComputeCosts(tbl, v, up_preds, up_scsrs)

        if __debug__:
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

        if __debug__:
            print("Processed vertex " + str(v) + "; Table size: " +
                    str(tbl.shape[0]) + "\n")
        return tbl

    def ProcessGraph(self):
        for v in self.SortNodes():
            tbl = self.ProcessVertex(v)

        assert(len(tbl.columns) == self.n_nodes + 1)
        return tbl

    def ProcessVerticesNaive(self, vs, tbl):
        vs_labels = list(str(v) for v in vs)
        next_vs = []
        self.processed_nodes.update(vs)
        self.processed_node_labels.update(vs_labels)

        # Add vertex and edge costs for vertices in 'vs
        vs_set = set(vs)
        for v in vs:
            tbl = MergeTables(tbl, self.CfgToDf(v))

        for v in vs:
            tbl = AddVertexCosts(v, self.vert_costs[v], tbl)

            # Add edge cost with predecessors 'p', only when 'p' has been
            # processed, and not part of 'vs'
            for p in self.G.predecessors(v):
                if p in self.processed_nodes:
                    if p not in vs_set:
                        tbl = AddEdgeCosts(p, v, self.edge_costs[(p, v)], tbl)
                else:
                    next_vs.append(p)

            # Add edge cost with processed successors, including those in 'vs'
            for s in self.G.successors(v):
                if s in self.processed_nodes:
                    tbl = AddEdgeCosts(v, s, self.edge_costs[(v, s)], tbl)
                else:
                    next_vs.append(s)

        if __debug__:
            print("Processing vertices " + str(vs) + "; Table size: " +
                    str(tbl.shape[0]))
        tbl = ReduceTable(tbl, vs_labels, 'costs')
        if __debug__:
            print("Processed vertices " + str(vs) + "; Table size: " +
                    str(tbl.shape[0]) + "\n")
        return next_vs, tbl

    def ProcessGraphNaive(self):
        q = [0]
        tbl = self.CfgToDf(0)
        while q:
            q, tbl = self.ProcessVerticesNaive(q, tbl)
        tbl = ReduceTable(tbl, None, 'costs')

        assert len(tbl.columns) == (self.n_nodes + 1)
        return tbl

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--procs", type=int, required=False, default=8,
            help="No. of processors.")
    parser.add_argument("-b", "--batch", type=int, required=False, default=128,
            help="Batch size.")
    parser.add_argument("-m", "--model", type=int, required=False, default=128,
            help="Model size.")
    parser.add_argument("-g", "--graph", type=str, required=False,
            choices=['alexnet', 'resnet101', 'inception3', 'rnnlm',
                'transformer'],
            default='alexnet', help="Neural net graph.")
    parser.add_argument('--flops', type=float, required=False, default=10.0,
            help='Peak FLOPS of each device in TFLOPS.')
    parser.add_argument('--bw', type=float, required=False, default=16.0,
            help='Peak inter-connection bandwidth in GBytes/sec')
    parser.add_argument("--profile", dest="profile", action='store_true',
            help="Turn on/off profiling.")
    parser.add_argument("--measure", dest="measure", action='store_true',
            help="Turn on/off measurement.")
    parser.add_argument("-d", "--dump-graph", dest="dump_graph",
            action='store_true', help="Dump the graph in dot format to the file "
            "graph.dot in the working directory.")
    parser.add_argument('--algo', type=int, required=False, default=0,
            choices=[0, 1], help='Algorithm to be used to compute strategy')
    parser.set_defaults(profile=False)
    args = vars(parser.parse_args())

    batch_size = args['batch']
    hidden_dim_size = args['model']
    n_procs = args['procs']

    # Profiling
    if args['profile']:
        pr = cProfile.Profile()

    # Create input graph
    G = graph.CreateGraph(args['graph'], batch_size, hidden_dim_size, n_procs,
            args['flops'], args['bw'])
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

    if args['measure']:
        warmup=2
        repeats=5
    else:
        warmup=0
        repeats=1

    if args['algo'] == 0:
        Process = lambda g: Processor(g).ProcessGraph()
    elif args['algo'] == 1:
        Process = lambda g: Processor(g).ProcessGraphNaive()
    else:
        assert False

    for i in range(warmup):
        g_tbl = Process(G)
    start = dt.datetime.now()
    for i in range(repeats):
        g_tbl = Process(G)
    end = dt.datetime.now()
    print("Processing time: ", (end-start)/float(repeats))
    print("")
    if args['profile']:
        pr.disable()
        pr.print_stats(sort='cumtime')

    # Convert 'g_tbl' into Series from DataFrame
    assert g_tbl.shape[0] == 1
    cols = []
    cost = 0
    for c in g_tbl.columns:
        try:
            cols.append(int(c))
        except ValueError:
            assert c.startswith('costs')
            cost = g_tbl.iloc[0][c]
            g_tbl.drop(c, 1, inplace=True)
    g_tbl.columns = cols
    g_tbl = g_tbl.iloc[0].sort_index()

    print("Strategy with minimum cost:")
    print("=====")
    print(g_tbl.to_string())
    print("=====")
    print("Total cost: %0.2f" % cost)


if __name__ == "__main__":
    main()

