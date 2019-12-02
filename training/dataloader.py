import networkx as nx
import numpy as np
import pandas as pd
import random

class Dataset:
	def __init__(self, directory = './data', embeddings=None, G=None, embeddings_file=None, graph_file=None):
		
		self.directory = directory

		if embeddings is None:
			assert(embeddings_file is not None)
			self.embeddings = load_embeddings(embeddings_file)
		else:
			self.embeddings = embeddings

		if G is None:
			assert(graph_file is not None)
			self.G = self.load_graph(graph_file)
		else:
			self.G = G

		pos_examples = self.get_positive_examples()
		self.num_pos_examples = len(pos_examples)
		neg_examples, self.edges_used = self.get_negative_examples(num_pos_examples)
		all_examples = pos_examples + neg_examples
		cols = ['src' + str(i) for i in range(embedding_dim)] + ['dst' + str(i) for i in range(embedding_dim)] + ['label']
		df = pd.DataFrame(all_examples, columns = cols)
		df.reset_index()
		temp_mask = np.random.rand(df.shape[0])
		train_mask = (temp_mask < .9)
		test_mask = (temp_mask >= .9)
		self.columns = df.columns
		self.train_df = df[train_mask]
		self.test_df = df[test_mask]

	def train_data(self):
		return self.train_df[self.columns[:-1]].values, self.train_df[self.columns[-1]].values

	def test_data(self):
		return self.test_df[self.columns[:-1]].values, self.test_df[self.columns[-1]].values

	def load_graph(self, filename):
	    df = pd.read_csv(self.directory + '/' + filename, header=None, names=['source', 'target', 'weight'])
	    G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=nx.Graph())
	    return G

	def load_embeddings(self, filename):
	    x = np.load(self.directory + filename, allow_pickle = True)
	    return x.item()

	# generate positive examples of edges
	def get_positive_examples(self):
	    pos_examples = []
	    for edge in self.G.edges():
	        src_embedding = self.embeddings[edge[0]]
	        dst_embedding = self.embeddings[edge[1]]
	        edge_vector = src_embedding + dst_embedding + [1] # label = 1
	        pos_examples.append(edge_vector)
	    return pos_examples

	# generate negative examples
	def get_negative_examples(self, num_examples, attempts = 3000000, len_threshold = 5):
	    node_list = list(G.nodes())
	    neg_examples = []
	    edges_used = set()
	    for i in range(attempts):
	        if len(neg_examples) == num_examples:
	            break
	        rnd_node_pair = random.choices(node_list, k = 2)
	        src = rnd_node_pair[0]
	        dst = rnd_node_pair[1]
	        if self.G.has_edge(src, dst):
	            continue
	        try:    
	            path_length = nx.shortest_path_length(self.G, source=src, target=dst, weight = None)
	        except nx.NetworkXNoPath:
	            continue
	        if(path_length) >= len_threshold:
	            src_embedding = self.embeddings[src]
	            dst_embedding = self.embeddings[dst]
	            edge_vector = src_embedding + dst_embedding + [0] # label = 0
	            neg_examples.append(edge_vector)
	            edges_used.add((src, dst))
	    return neg_examples, edges_used


	   # generate inference examples
	def get_inference_examples(self, num_examples = 100000, attempts = 1000000):
	    node_list = list(self.G.nodes())
	    inference_examples = []
	    for i in range(attempts):
	        if len(inference_examples) == num_examples:
	            break
	        rnd_node_pair = random.choices(node_list, k=2)
	        src = rnd_node_pair[0]
	        dst = rnd_node_pair[1]
	        if self.G.has_edge(src, dst):
	            continue
	        edge_tuple = (src, dst)
	        if edge_tuple not in self.edges_used:
	            src_embedding = self.embeddings[src]
	            dst_embedding = self.embeddings[dst]
	            edge_vector = src_embedding + dst_embedding
	            inference_examples.append(edge_vector)
	    return np.vstack(inference_examples)
	







