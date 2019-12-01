import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community as nxcm
import copy
import snap
import community

def louvain(G):
    partition = community.best_partition(G)
    return partition

def modularity(partition):
    modularity = community.modularity(partition, G)
    return modularity

def plot_hist(partition,filename='community_hist.png'):
    communities_u = {}
    for v in partition.values():
        communities_u[v] = communities_u.get(v, 0) + 1

    plt.figure()
    plt.hist(communities_u.values())
    plt.savefig(filename)
    
def communityGN(graph):
    communities_generator = nxcm.girvan_newman(graph)
    top_level_communities = next(communities_generator)
    partitions = [list(c) for c in top_level_communities]
    layer = []
    for partition in partitions:
        subgraph = graph.subgraph(partition)
        layer.append(subgraph)
    return layer

#Identification
"""
(1) Identify a layer of communities via the base method;
(2) Weaken the structure of the detected layer;
(3) Repeat until the appropriate number of layers are found
"""

def identifyLayer(graph, algorithm='GN'):
    if algorithm == 'GN':
        return communityGN(graph)
    else:
        raise Exception("Invalid argument for community detection algorithm, %s" % algorithm)

def removeEdge(graph,layer):
    #Take layer, a division of graph into communities, and 
    #weaken the communities within graph by removing the edges
    #in that layer completely
    relaxed_graph = graph.copy()
    for subgraph in layer:
        relaxed_graph.remove_edges_from(subgraph.edges)
    return relaxed_graph

def reduceWeight(graph,layer):
    """
    Take layer, a division of graph into communities, and 
    weaken the communities within graph by reducing the 
    weight of each edge within community C_k by a factor of
    q'k = p_k/q_k
    q_k = (d_k - 2e_kk)/(n_k(n-n_k)
    p_k = e_kk/(0.5n_k*(n_k-1))
    Where p_k is the observed edge probability of Community C_k
    q_k is the background block probability
    n_k is the number of nodes in C_k
    d_k is the sum of degrees of nodes in C_k
    e_kk is the number of edges in C_k
    n is the total number of nodes in graph
    """
    pass

