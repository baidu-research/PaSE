import networkx as nx
import numpy as np
from functools import partial
import operator

import cost as cst
import config as cfg


class Domain():
    def __init__(self, dim):
        self.sz = dim;
        self.m = dim[0];
        self.n = dim[1];
        self.k = dim[2];


def AssignCostsToNode(node_attr, n_procs):
    node_dom = node_attr['dom']

    configs = cfg.GetConfigs([len(node_dom.sz)], n_procs)
    node_attr['configs'] = configs

    costs = []
    for set_config in configs:
        set_cost = []
        for config in set_config:
            set_cost.append(cst.CompCost(node_dom.sz, config))
        costs.append(max(set_cost))

    assert(len(configs) == len(costs))
    node_attr['costs'] = costs


def AssignCostsToEdges(G, edge, n_procs):
    nodes = G.nodes(data=True)
    src_attr = nodes[edge[0]]
    tgt_attr = nodes[edge[1]]
    edge_attr = edge[2]

    src_dom = src_attr['dom']
    tgt_dom = tgt_attr['dom']

    src_configs = src_attr['configs']
    tgt_configs = tgt_attr['configs']

    configs = []
    costs = []
    for i in src_configs:
        for j in tgt_configs:
            configs.append((i, j))
            cost = cst.CommCost(src_dom, tgt_dom, Domain(i[0]), Domain(j[0]))
            costs.append(cost)

    edge_attr['configs'] = configs
    edge_attr['costs'] = costs


def EliminatePaths(G):
    # Get nodes with one in and out edges
    single_degree_nodes = []
    for in_deg, out_deg in zip(G.in_degree(), G.out_degree()):
        assert(in_deg[0] == out_deg[0])
        if in_deg[1] <= 1 and out_deg[1] <= 1:
            single_degree_nodes.append(in_deg[0])

    # Include predecessor and successor of single degree nodes to the list
    neighbors = []
    for node in single_degree_nodes:
        neighbors += list(G.predecessors(node))
        neighbors += list(G.successors(node))

    single_degree_nodes += neighbors
    path_graph = G.subgraph(single_degree_nodes)
    print(G.edges())
        


def main():
    batch_size = 1024
    hidden_dim_size = 512
    n_procs = 8

    G = nx.DiGraph()
    G.add_node(1, dom=Domain([batch_size, hidden_dim_size, hidden_dim_size]))
    G.add_node(2, dom=Domain([batch_size, hidden_dim_size, hidden_dim_size]))
    G.add_node(3, dom=Domain([batch_size, hidden_dim_size, hidden_dim_size]))
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    # Assign configs and costs for each node
    for _, attr in G.nodes(data=True):
        AssignCostsToNode(attr, n_procs)
        #print(attr['costs'])

    # Assign configs and costs for each edge
    for e in G.edges(data=True):
        AssignCostsToEdges(G, e, n_procs)
        #for i,j in zip(e[2]['configs'], e[2]['costs']):
        #    print((i,j))

    # Eliminate straight-line paths
    EliminatePaths(G)


if __name__ == "__main__":
    main();

