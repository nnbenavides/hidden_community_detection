import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community as nxcm
import copy
import pandas as pd
import numpy as np
import snap
import community
import random

#layer is an array of subgraphs of G
def layer_to_partition(layer,G):
    partition = {}
    for i,subgraph in enumerate(layer):
        for node in list(subgraph.nodes):
            partition[node] = i
    return partition


#partition is a dictionary of node:community mappings
def partition_to_layer(partition,G):
    max_value = max(partition.values())
    nodes = [[] for v in range(max_value+1)]
    for key,val in partition.items():
        nodes[val].append(key)
    layer = []
    for nodelist in nodes:
        layer.append(G.subgraph(nodelist))
    return layer


def loadGraphs():
    df = pd.read_csv('data/reddit_nodes_weighted_full.csv', header=None, names=['source', 'target', 'weight'])
    G_weighted = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=nx.Graph())
    G = nx.read_edgelist('data/undirected_unweighted_union_reddit_edge_list.tsv')
    return G, G_weighted


def louvain(G):
    partition = community.best_partition(G)
    return partition

def modularity(partition,G):
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
    #next_level_communities = next(communities_generator)
    partitions = [list(c) for c in top_level_communities]
    layer = []
    for partition in partitions:
        subgraph = graph.subgraph(partition)
        #print(list(subgraph.nodes))
        #print(list(subgraph.edges))
        layer.append(subgraph)
    # print(communities)
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

def get_q_prime_k(subgraph,graph,layer):
    n_k = len(subgraph)
    n = len(graph)
    e_kk = subgraph.number_of_edges()
    d_k = sum([d for n,d in graph.degree(subgraph.nodes)])
    if n_k*(n_k - 1) == 0:
        p_k = e_kk
    else:
        p_k = 2*e_kk/(n_k*(n_k - 1))
    if n_k*(n - n_k) == 0:
        q_k = (d_k - 2*e_kk)/n
        print("n-n_k or n_k is 0")
    else:
        q_k = (d_k - 2*e_kk)/(n_k*(n - n_k))
    q_prime_k = q_k/p_k
    return q_prime_k

def reduceEdge(graph,layer):
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
    relaxed_graph = graph.copy()
    for subgraph in layer:
        q_prime_k = get_q_prime_k(subgraph,graph,layer)
        edges_to_remove = [edge for edge in subgraph.edges if random.random() < (1-q_prime_k)]
        relaxed_graph.remove_edges_from(edges_to_remove)
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
    reduce weight of
    """
    relaxed_graph = graph.copy()
    for subgraph in layer:
        q_prime_k = get_q_prime_k(subgraph,graph,layer)
        #print("q_prime_k",q_prime_k)
        for u, v, weight in relaxed_graph.edges.data('weight'):
            #print(q_prime_k,relaxed_graph.edges[u, v]['weight'])
            relaxed_graph.edges[u, v]['weight'] *= (1+q_prime_k)
    return relaxed_graph


#Refinement
"""
(1) Weaken the structures of all other layers from the original
network to obtain a reduced network;
(2) Apply the base algorithm to the resulting network.

(1) Calculate Q0 for t = 0, i.e. aer identication, before any
renement is conducted;
(2) Perform T = 10 tentative iterations of renement, and
calculate Qt for each t ∈ {1, ...,T };
(3) Calculate the average improvement ratio of modularity per
iteration4
: RT = Sum Q_t from 1 to T/ (10*Q_0)
"""
def average_modularity(layers,G):
    Q = 0
    for layer in layers:
        #print(layer)
        partition = layer_to_partition(layer,G)
        m = modularity(partition,G)
        Q += m
    Q /= len(layers)
    return Q

def refinement(layers,G):
    output_layers = []
    Q = 0
    for i,layer in enumerate(layers):
        G_reduced = G.copy()
        for j in range(len(layers)):
            if j == i:
                continue
            G_reduced = reduceWeight(G_reduced,layers[j])
        partition = louvain(G_reduced)
        m = modularity(partition,G_reduced)
        output_layers.append(partition_to_layer(partition,G_reduced))
    return output_layers, G_reduced

def hicode(G, num_layers = 1):
    #Identification
    layers = []
    G_curr = G.copy()
    partition = louvain(G_curr)
    layer = partition_to_layer(partition,G_curr)
    layers.append(layer)
    for i in range(num_layers-1):
        print("iteration: ",i)
        G_curr = reduceWeight(G_curr,layer)
        partition = louvain(G_curr)
        layer = partition_to_layer(partition,G_curr)
        layers.append(layer)
    Q_0 = average_modularity(layers,G_curr)

    #refinement
    print("Refinement")
    R_t = 0
    for i in range(10):
        print(i)
        layers, G_curr = refinement(layers,G_curr)
        Q = average_modularity(layers,G_curr)
        R_t += Q

    R_t /= 10*Q_0

    return R_t, G_curr
