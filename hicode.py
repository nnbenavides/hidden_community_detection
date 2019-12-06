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

R_t = utils.hicode(G,1)
print(R_t)
