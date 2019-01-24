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


# Returns the config that provides minimum cost for its neighboring
# configurations  'costs'
# neighbors: List of neighbors of 'node'
# node_costs: 2D array, with each row having a configuration and its cost
# neigh_costs: List of configuration and cost. Each item in the list corresponds to
#              a single config and its cost for the corresponding item in 'neighbors'
def GetMinConfig(G, node, neighbors, node_costs, neigh_costs):
    # TODO: Handle this case
    if len(neighbors) == 0:
        return

    assert(len(neighbors) == len(neigh_costs))

    neigh_costs_array = np.array(neigh_costs)
    assert(neigh_costs_array.ndim == 2)

    # Get the sum of computation costs of neighboring nodes
    # 'min_cost' is scalar
    min_cost = np.sum(neigh_costs_array[:, -1])

    # Add 'min_cost' with costs for different configs of 'node'
    # min_cost is a 2D array, each row is a node config and corresponding total
    # comp cost
    cost = node_costs
    cost[:,-1] += min_cost
    min_cost = cost

    neigh_cost = dict(zip(neighbors, neigh_costs))

    # Iterate over predecessors of 'node' and add communication cost
    for pred, _, edge_cost in G.in_edges(node, data='costs'):
        assert(pred in neigh_cost)

        # Get the config-cost array of the source vertex of the edge
        pred_cost = neigh_cost[pred]
        assert(pred_cost.ndim == 1)

        # Extract the rows in 'edge_cost' whose source vertex config matches the
        # config in 'pred_cost'
        idx = pred_cost.shape[0] - 1
        cost = edge_cost[np.equal(edge_cost[:,:idx],
            pred_cost[:-1]).all(axis=1)]

        assert(cost.shape[0] == min_cost.shape[0])
        assert(np.equal(cost[:, idx:-1], node_costs[:, :-1]).all() == True)
        assert(np.equal(cost[:, idx:-1], min_cost[:, :-1]).all() == True)

        # Add the comm cost 'cost' with comp cost 'min_cost'
        min_cost[:,-1] += cost[:,-1]

    # Iterate over successors of 'node' and add communication cost
    for _, succ, edge_cost in G.out_edges(node, data='costs'):
        assert(succ in neigh_cost)

        # Get the config-cost array of the source vertex of the edge
        succ_cost = neigh_cost[succ]
        assert(succ_cost.ndim == 1)

        # Extract the rows in 'edge_cost' whose source vertex config matches the
        # config in 'succ_cost'
        idx = node_costs.shape[1] - 1
        cost = edge_cost[np.equal(edge_cost[:, idx:-1],
            succ_cost[:-1]).all(axis=1).nonzero()]

        assert(cost.shape[0] == min_cost.shape[0])
        assert(np.equal(cost[:, :idx], node_costs[:, :-1]).all() == True)
        assert(np.equal(cost[:, :idx], min_cost[:, :-1]).all() == True)

        # Add the comm cost 'cost' with 'min_cost'
        min_cost[:,-1] += cost[:,-1]

    # Get the config and cost corresponding to minimum cost
    min_idx_val = min_cost[min_cost[:,-1].argmin(axis=0)]
    min_config = min_idx_val[:-1]

    # TODO: Set DP attribute to node with neighbor configs and min_config

    ## Reset edge configs to contain only min_config rows
    #for pred, _, attr in G.in_edges(node, data=True):
    #    tbl = attr['costs']

    #    pred_config = neigh_cost[pred][:-1]
    #    idx = pred_config.shape[0]

    #    # Only include rows whose source configs are different from
    #    # 'pred_config', or target config is same as 'min_config'
    #    cond1 = np.not_equal(tbl[:,:idx], pred_config).all(axis=1)
    #    cond2 = np.equal(tbl[:,idx:-1], min_config).all(axis=1)
    #    cond = np.logical_or(cond1, cond2)
    #    attr['costs'] = tbl[cond]

    #for _, succ, attr in G.out_edges(node, data=True):
    #    tbl = attr['costs']

    #    succ_config = neigh_cost[succ][:-1]
    #    idx = succ_config.shape[0]

    #    # Only include rows whose source configs are different from
    #    # 'succ_config', or target config is same as 'min_config'
    #    cond1 = np.not_equal(tbl[:,idx:-1], succ_config).all(axis=1)
    #    cond2 = np.equal(tbl[:,:idx], min_config).all(axis=1)
    #    cond = np.logical_or(cond1, cond2)
    #    attr['costs'] = tbl[cond]


def Process(G, node, neighbors, processed_nodes):
    node_attrs = G.nodes(data=True)

    unprocessed_neighbors = [i for i in neighbors if i not in processed_nodes]
    node_attrs[node]['unprocessed_neigh'] = unprocessed_neighbors

    # Get the 'costs' array of each neighbor node
    iterators = []
    for n in neighbors:
        iterators.append(node_attrs[n]['costs'])

    # Iterate over all combinations of costs
    node_cost = node_attrs[node]['costs']
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

