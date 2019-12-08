import utils
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


G, G_weighted = utils.loadGraphs()

R_t, G_2, layers_2 = utils.hicode(G_weighted,2)
print("hicode 2",R_t)
nx.write_weighted_edgelist(G_weighted, 'results/G2_weighted.edgelist')

nodes_mapping = utils.load_nodes_mapping()
partitions = []
for num_layer, layer in enumerate(layers_2):
    community_count = 0
    for i,subgraph in enumerate(layer):
        if len(subgraph.nodes) > 100:
            community_count += 1
    print("community_count",community_count)
    partition = utils.layer_to_partition(layer,G_weighted)
    print("layer number ", num_layer + 1)
    print(utils.modularity(partition,G_weighted))
    print(utils.modularity(partition,G_2))
    partitions.append(partition)
    reverse_comms = utils.reverse_dict(partition)
    utils.write_results_to_file(reverse_comms,nodes_mapping,"layer_"+str(num_layer + 1))
