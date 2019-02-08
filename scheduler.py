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
        idx = pd.Index(configs, dtype=tuple)

        attr['config_tuples'] = config_tuples
        attr['configs'] = configs
        attr['costs'] = pd.Series(costs, index=idx)


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
        idx = pd.MultiIndex.from_product([src_config_tuples, tgt_config_tuples])
        edge_attr['costs'] = pd.Series(costs, index=idx)


def CreateGraph(batch_size, hidden_dim_size):
    G = nx.DiGraph()
    G.add_node(1, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(2, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_node(3, dim=3,dom=[batch_size, hidden_dim_size, hidden_dim_size])
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    return G


#def GetEdgeCosts(G, v, v_tbl, n, neigh_tbl):
#    edges = G.edges()
#    edge_configs = nx.get_edge_attributes(G, 'configs')
#    edge_costs = nx.get_edge_attributes(G, 'costs')
#
#    if (v, n) in edges:
#        tbl = np.concatenate(RowCartesian(v_tbl, neigh_tbl), axis=1)
#        idx = np.searchsorted(edge_configs[(v,n)], tbl)
#        costs = edge_costs[(v,n)][idx]
#    else:
#        assert((n, v) in G.edges())


def ExtendTable(tbl, verts, vert_cfgs):
    cols = tbl.columns

    # If all the vertices are already present, just return the original table
    if set(verts).issubset(cols):
        return tbl

    tbl_with_key = tbl.assign(key=0)
    for v in verts:
        if v not in cols:
            v_df = pd.DataFrame(pd.Series(vert_cfgs[v]), columns=[v])
            v_df = v_df.assign(key=0)
            if tbl.empty:
                tbl_with_key = v_df
            else:
                tbl_with_key = pd.merge(left=tbl_with_key, right=v_df, on
                        ='key')

    return tbl_with_key.drop('key', 1)


def GetVertexIndices(v, g_tbl, vert_costs):
    if v in g_tbl.columns:
        in_tbl = True
        v_idx = pd.Index(g_tbl.loc[:, v])
    else:
        in_tbl = False
        v_idx = vert_costs[v].index

    return in_tbl, v_idx


def GetCosts(src, tgt, src_in_tbl, tgt_in_tbl, src_idx, tgt_idx, edge_costs,
        costs):
    if (not src_in_tbl) or (not tgt_in_tbl):
        m_idx = pd.MultiIndex.from_product([src_idx, tgt_idx])
    else:
        m_idx = pd.MultiIndex.from_tuples(zip(src_idx, tgt_idx))

    curr_costs = edge_costs.loc[m_idx]

    return costs


def ProcessVertex(G, v):
    g_tbl = G.graph['tbl']

    vert_cfgs = nx.get_node_attributes(G, 'config_tuples')
    vert_costs = nx.get_node_attributes(G, 'costs')
    edge_costs = nx.get_edge_attributes(G, 'costs')

    verts = [i for i in itertools.chain(G.predecessors(v), G.successors(v))]
    #verts.append(v)

    g_tbl = ExtendTable(g_tbl, verts, vert_cfgs)
    G.graph['tbl'] = g_tbl

    #v_in_tbl, v_idx = GetVertexIndices(v, g_tbl, vert_costs)
    #costs = pd.DataFrame(vert_costs[v].loc[v_idx], columns=['cost'])
    #costs.index.names = [v]

    #for n in G.predecessors(v):
    #    n_in_tbl, n_idx = GetVertexIndices(n, g_tbl, vert_costs)
    #    costs = GetCosts(n, v, n_in_tbl, v_in_tbl, n_idx, v_idx, edge_costs,
    #            costs)

    #for n in G.successors(v):
    #    n_in_tbl, n_idx = GetVertexIndices(n, g_tbl, vert_costs)
    #    GetCosts(v, n, v_in_tbl, n_in_tbl, v_idx, n_idx, edge_costs)

    #neigh = list(G.predecessors(v)) + list(G.successors(v))
    #names_idx = []
    #no_tbl_neigh = []
    #if names:
    #    for n in neigh:
    #        try:
    #            names_idx.append(names.index(n))
    #        except ValueError:
    #            no_tbl_neigh.append(n)
    #else:
    #    no_tbl_neigh = neigh

    ## Create a list of iterators
    #t = tbl[:, names_idx]
    #neigh_tbls = []
    #if t.size > 0:
    #    neigh_tbls = [t]
    #for n in no_tbl_neigh:
    #    neigh_tbls.append(vert_cfgs[n])

    ## Iterate over all sub-strategies
    #v_cfg = vert_cfgs[v]
    #for ss in itertools.product(*neigh_tbls):
    #    ss = np.concatenate(ss).reshape(1, -1)
    #    ss = np.repeat(ss, v_cfg.shape[0], axis=0)
    #    ss = np.concatenate((ss, v_cfg), axis=1)


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

    #print(G.graph['tbl'])


if __name__ == "__main__":
    main()

