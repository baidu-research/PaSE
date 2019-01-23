import networkx as nx
import numpy as np
from functools import partial
import operator
from sortedcontainers import SortedList
import itertools

import cost_np as cst
import config as cfg


def AssignCostsToNodes(G, n_procs):
    for _, attr in G.nodes(data=True):
        dom = attr['dom']
        configs = np.array(cfg.GetNodeConfigs(dom, n_procs))
        costs = cst.GetCompCosts(np.array(dom), configs)
        attr['configs'] = configs
        attr['costs'] = np.column_stack((configs, costs))


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

        edge_attr['src_configs'] = src_configs
        edge_attr['tgt_configs'] = tgt_configs
        edge_attr['costs'] = np.column_stack((src_configs, tgt_configs, costs))


def CreateGraph(batch_size, hidden_dim_size):
    G = nx.DiGraph()
    G.add_node(1, dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(2, dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(3, dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    return G


def GetMinConfig(G, node, neighbors, node_costs, costs):
    # TODO: Handle this case
    if len(neighbors) == 0:
        return

    assert(len(neighbors) == len(costs))

    consolidated_costs = np.array(costs)
    assert(consolidated_costs.ndim == 2)

    # Get the sum of computation costs of neighboring nodes
    min_cost = np.sum(consolidated_costs[:, -1])

    # Add 'min_cost' with different configs of 'node'
    min_cost += node_costs

    # Iterate over predecessors of 'node' and add communication cost
    neigh_cost = dict(zip(neighbors, costs))
    for pred, _, edge_cost in G.in_edges(node, data='costs'):
        assert(pred in neigh_cost)
        pred_cost = neigh_cost[pred]
        assert(pred_cost.ndim == 1)

        cost = edge_cost[np.equal(edge_cost[:,:pred_cost.shape[0]-1],
            pred_cost[:-1]).all(axis=1).nonzero()]
        print(cost)

    # Iterate over each configuration of 'node'
    #for n, tbl in zip(neighbors, costs):
    #    curr_cost += tbl[-1]
        #config = tbl[0]
        #idx = 0
        #if G.has_edge(node, n):
        #    idx = 1
        #else:
        #    assert(G.has_edge(n, node))
            


def Process(G, node, neighbors, processed_nodes):
    node_costs = G.nodes(data='costs')

    # Get the 'costs' array of each neighbor node
    iterators = []
    for n in neighbors:
        iterators.append(node_costs[n])

    # Iterate over all combinations of costs
    node_cost = node_costs[node]
    for vals in itertools.product(*iterators):
        GetMinConfig(G, node, neighbors, node_cost, vals)


def DP(G):
    n_nodes = G.number_of_nodes()
    min_key, min_val = 0, n_nodes
    unprocessed_neighbors = {}

    # Set the unprocessed neighbor count of each node to its node degree
    for k, v in nx.degree(G):
        assert(v > 0)
        unprocessed_neighbors[k] = v
        if v < min_val:
            min_key, min_val = k, v

    # Iterate over unprocessed nodes
    processed = SortedList()
    while len(unprocessed_neighbors) > 0:
        # Get the node with minimum unprocessed neighbors
        min_node = min(unprocessed_neighbors, key=unprocessed_neighbors.get)
        del unprocessed_neighbors[min_node]
        assert(min_node not in processed)

        # Process the node
        neighbors = list(G.predecessors(min_node)) + list(G.successors(min_node))
        Process(G, min_node, neighbors, processed)
        processed.add(min_node)

        # Update the unprocessed neighbor count of 'min_node's neighbors
        for node in neighbors:
            if node in unprocessed_neighbors:
                unprocessed_neighbors[node] -= 1

    assert(len(processed) == n_nodes)



def main():
    batch_size = 32
    hidden_dim_size = 32
    n_procs = 4

    # Create input graph
    G = CreateGraph(batch_size, hidden_dim_size)

    # Assign config list to nodes in 'G', and their costs
    AssignCostsToNodes(G, n_procs)

    # Assign configs and costs for each edge
    AssignCostsToEdges(G, n_procs)

    # Run the DP
    DP(G)


if __name__ == "__main__":
    main()

