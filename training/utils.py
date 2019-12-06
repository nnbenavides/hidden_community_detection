import numpy as np
from random import choice

def node2vec_embedder(embedding_file):
	embeddings = {}
	f = open(embedding_file, 'r')
	f.readline()
	line = f.readline()
	line = f.readline()
	while(line):
		if line[0] == ' ': line = line[1:]
		line = line.split(' ')
		# key = int(line[0])
		embedding = [float(x) for x in line[1:]]
		embeddings[str(int(line[0]))] = embedding
		line = f.readline()
		
	return embeddings


def make_filepath(args):
	print(args["num_walks"])
	print(args["walk_length"])
	print(args["window"])
	node2vecstr = ('_numwalks-%d_walklength-%d_window-%d' % (args["num_walks"], args["walk_length"], args["window"])) if args["embedder"].lower()=='node2vec' else ''
	embedd_str = '-dimension-%d_lr-%.4f_seed-%d_epochs-%d%s' % (args["embedding_dim"], args["embedding_lr"], args["embedding_seed"], args["embedding_epochs"], node2vecstr) if args["embedder"].lower() != 'rolx' else ''
	layer_str = ''.join([str(d)+'+' for d in args["layers"]])
	nn_str = ('dense_' if args["dense_classifier"] else 'rnn_') + ('dropout-%.2f_' % args["dropout"] if args["dropout"] is not None else '') + 'layers-%s' % layer_str[:-1]
	full_filepath = '%s%s_NN-%s' % (args["embedder"].lower(), embedd_str, nn_str)

	return full_filepath, embedd_str



arguments = ['embedder', 'embedding_dim', 'embedding_seed', 'embedding_lr', 'walk_length', 'num_walks', 'window', 'p', 'q']
embed_args = [["rolx"],#["node2vec", "line", "rolx"],
		[32, 64, 96, 128, 256, 512],
		[1234, 4321],
		[0.001, 0.005, 0.01, 0.05, 0.0001]]

node2vecs = [[10, 20, 30, 40, 50], 
			  [20, 40, 60, 80, 100, 150, 200], 
			  [5, 10, 20], 
			  [.2, .4, .6, .8, 1.0], 
			  [.2, .4, .6, .8, 1.0]]
# embed_args = [['node2vec', 'line'],
#       [15, 24, 32, 64, 96, 128, 256, 512, 1024],
#       [1234, 4321],
#       [0.001, 0.005, 0.01, 0.05, 0.0001]]

# node2vecs = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
#               [20, 40, 60, 80, 100, 150, 200, 300, 400], 
#               [5, 10, 20, 40, 60, 80, 100], 
#               [.2, .4, .6, .8, 1.0], 
#               [.2, .4, .6, .8, 1.0]]

nn_args = [[False, True, True, True],[None, .25, .5]]
layer_choices = [[2, 3, 4, 5, 6, 8], [1, 2, 3]]

def create_args(directory='./data', graph_file='reddit_nodes_weighted_full.csv', embedding_batch_size=1024, embedding_epochs=250):
	embedder = choice(embed_args[0])

	embedding_dim = 96 if embedder == "rolx" else choice(embed_args[1])
	embedding_seed = choice(embed_args[2])
	embedding_lr = choice(embed_args[3])
	p = 1.0 if embedder != "node2vec" else choice(node2vecs[3])
	q = 1.0 if embedder != "node2vec" else choice(node2vecs[4])
	walk_length = 10 if embedder != "node2vec" else node2vecs[0]
	num_walks = 10 if embedder != "node2vec" else node2vecs[1]
	window = 10 if embedder != "node2vec" else node2vecs[2]
	workers = 1
	dense = choice(nn_args[0])
	dropout = choice(nn_args[1]) if dense else None
	layers = choice(layer_choices[0] if dense else layer_choices[1])
	layers = gen_layers(layers, dense)
	patience=10
	validation_split=0.2
	batch_size=120
	epochs=1000
	temp_folder='temp_folder'
	args = (directory, embedder, graph_file, embedding_batch_size, embedding_epochs, embedding_dim, 
		embedding_seed, embedding_lr, p, q, walk_length, num_walks, window, workers, dropout, layers, dense, 
		patience, validation_split, batch_size, epochs, temp_folder, )

	return args



layer1 = [32, 64, 128, 256, 512]

def gen_layers(layers, dense):
	
	start = choice(layer1)
	while(dense and layers > 4 and start < 256):
		start = choice(layer1)

	full = [start]
	for i in range(layers):
		if (not dense) and i == (layers-1):
			# if recurrent manually add the very last layer
			full.append(full[-1] if full[-1] < 32 else 32)
			break

		full.append(start)
		# randomly decide to half the next layers input unit size
		if start > 16 and np.random.rand() > .5:
			start = int(start/2)

	return full