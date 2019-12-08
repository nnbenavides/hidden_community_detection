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

R_t, G_out = utils.hicode(G_weighted,1)
print("hicode 1",R_t)

R_t, G_out = utils.hicode(G_weighted,2)
print("hicode 2",R_t)

R_t, G_out = utils.hicode(G_weighted,3)
print("hicode 3",R_t)
