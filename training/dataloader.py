import networkx as nx
import numpy as np
import pandas as pd
import random

class Dataset:
	def __init__(self, directory = './data', embeddings=None, G=None, embeddings_file=None, graph_file=None, embedding_dim=64):
		
		self.directory = directory

		if embeddings is None:
			# assert(embeddings_file is not None)
			self.embeddings = self.load_embeddings(embeddings_file)
		else:
			self.embeddings = embeddings

		if G is None:
			# assert(graph_file is not None)
			self.G = self.load_graph(graph_file)
		else:
			self.G = G

		self.weights_dict = self.get_weights_dict(directory+'/' + graph_file)
		# print("got weight dict")
		pos_examples = self.get_positive_examples()
		# print("got positive exampels")
		self.num_pos_examples = len(pos_examples)
		neg_examples, self.edges_used = self.get_negative_examples(20)#self.num_pos_examples)
		all_examples = np.vstack([pos_examples, neg_examples])
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

	def get_weights_dict(self, filename):
	    weights = pd.read_csv(filename, header = None)
	    weights.columns = ['src', 'dst', 'weight']
	    
	    weights_dict = {}
	    for i in range(weights.shape[0]):
	        src = weights.iloc[i, 0]
	        dst = weights.iloc[i, 1]
	        weight = weights.iloc[i, 2]

	        weights_dict[(src, dst)] = weight
	        weights_dict[(dst, src)] = weight

	    return weights_dict

	def load_embeddings(self, filename):
	    x = np.load(self.directory + filename, allow_pickle = True)
	    return x.item()

	# generate positive examples of edges
	def get_positive_examples(self):
	    pos_examples = []
	    for edge in self.G.edges():
	        if (str(edge[0]) not in self.embeddings) or (str(edge[1]) not in self.embeddings): continue
	        src_embedding = self.embeddings[str(edge[0])]
	        dst_embedding = self.embeddings[str(edge[1])]
	        edge_vector = src_embedding + dst_embedding + [self.weights_dict[(edge[0], edge[1])]] # label = edge weight
	        pos_examples.append(edge_vector)
	    return np.vstack(pos_examples)

	# generate negative examples
	def get_negative_examples(self, num_examples, attempts = 3000000, len_threshold = 5):
	    node_list = list(self.G.nodes())
	    neg_examples = []
	    edges_used = set()
	    for i in range(attempts):
	        if len(neg_examples) == num_examples:
	            break
	        rnd_node_pair = random.choices(node_list, k = 2)
	        src = rnd_node_pair[0]
	        dst = rnd_node_pair[1]
	        if (str(src) not in self.embeddings) or (str(dst) not in self.embeddings): continue
	        if self.G.has_edge(src, dst):
	            continue
	        try:    
	            path_length = nx.shortest_path_length(self.G, source=src, target=dst, weight = None)
	        except nx.NetworkXNoPath:
	            continue
	        if(path_length) >= len_threshold:
	            src_embedding = self.embeddings[str(src)]
	            dst_embedding = self.embeddings[str(dst)]
	            edge_vector = src_embedding + dst_embedding + [0] # label = 0
	            neg_examples.append(edge_vector)
	            edges_used.add((src, dst))
	    return np.vstack(neg_examples), edges_used


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
	        if (str(src) not in self.embeddings) or (str(dst) not in self.embeddings): continue
	        if self.G.has_edge(src, dst):
	            continue
	        edge_tuple = (src, dst)
	        if edge_tuple not in self.edges_used:
	            src_embedding = self.embeddings[src]
	            dst_embedding = self.embeddings[dst]
	            edge_vector = src_embedding + dst_embedding
	            inference_examples.append(edge_vector)
	    return np.vstack(inference_examples)
	







