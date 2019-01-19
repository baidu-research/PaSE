import networkx as nx
import numpy as np
from functools import partial
import operator

import cost_np as cst
import config as cfg


def AssignCostsToNodes(G, n_procs):
    for _, attr in G.nodes(data=True):
        dom = attr['dom']
        configs = np.array(cfg.GetNodeConfigs(dom, n_procs))
        costs = cst.GetCompCosts(np.array(dom), configs)
        print(configs.shape)
        attr['configs'] = configs
        attr['costs'] = costs


def AssignCostsToEdges(G, n_procs):
    nodes = G.nodes(data=True)

    for src, tgt, edge_attr in G.edges(data=True):
        src_attr = nodes[src]
        tgt_attr = nodes[tgt]

        src_dom = np.array(src_attr['dom'])
        tgt_dom = np.array(tgt_attr['dom'])

        src_configs = src_attr['configs']
        tgt_configs = tgt_attr['configs']

        # Repeat configs to get a cross-product of src and tgt configs
        orig_src_rows = src_configs.shape[0]
        orig_tgt_rows = tgt_configs.shape[0]
        src_configs = np.repeat(src_configs, repeats=orig_tgt_rows, axis=0)
        tgt_configs = np.tile(tgt_configs, (orig_src_rows, 1))
        assert(src_configs.shape == tgt_configs.shape)

        costs = cst.GetCommCosts(src_dom, tgt_dom, src_configs, tgt_configs)
        print(costs.shape)

        edge_attr['src_configs'] = src_configs
        edge_attr['tgt_configs'] = tgt_configs
        edge_attr['costs'] = costs


def CreateGraph(batch_size, hidden_dim_size):
    G = nx.DiGraph()
    G.add_node(1, dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(2, dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(3, dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    return G


def main():
    batch_size = 1024
    hidden_dim_size = 512
    n_procs = 8

    # Create input graph
    G = CreateGraph(batch_size, hidden_dim_size)

    # Assign config list to nodes in 'G', and their costs
    AssignCostsToNodes(G, n_procs)

    # Assign configs and costs for each edge
    AssignCostsToEdges(G, n_procs)


if __name__ == "__main__":
    main();

