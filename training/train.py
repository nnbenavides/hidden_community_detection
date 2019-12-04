from node2vec import Node2Vec
from ge import LINE
import networkx, os, argparse, json
from dataloader import Dataset
from model import Classifier
from utils import node2vec_embedder, make_filepath
from tensorflow import set_random_seed
import numpy as np
import tensorflow as tf
import pandas as pd
import networkx as nx

np.random.seed(2019)
tf.compat.v1.set_random_seed(2019)


parser = argparse.ArgumentParser(description='args')
parser.add_argument('--directory', dest='directory', type=str, default='./data')
parser.add_argument('--embedder', dest='embedder', type=str, default='node2vec')
parser.add_argument('--graph_file', dest='graph_file', type=str, default='reddit_nodes_weighted_full.csv')
parser.add_argument('--embedding_batch_size', dest='embedding_batch_size', type=int, default=1024)
parser.add_argument('--embedding_epochs', dest='embedding_epochs', type=int, default=250)
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=96)
parser.add_argument('--embedding_seed', dest='embedding_seed', type=int, default=2019)
parser.add_argument('--embedding_lr', dest='embedding_lr', type=float, default=0.05)
parser.add_argument('--q', dest='q', type=float, default=1.0)
parser.add_argument('--p', dest='p', type=float, default=1.0)
parser.add_argument('--walk_length', dest='walk_length', type=int, default=50)
parser.add_argument('--num_walks', dest='num_walks', type=int, default=200)
parser.add_argument('--window', dest='window', type=int, default=10)
parser.add_argument('--workers', dest='workers', type=int, default=8)
parser.add_argument('--dropout', dest='dropout', type=float, default=None)
parser.add_argument('--layers', dest='layers', nargs='+', help='space seperated list which specifies size of each layer, if using rnn the last value is the value for the dense network')
parser.add_argument('--dense_classifier', dest='dense_classifier', type=int, default=1)
parser.add_argument('--patience', dest='patience', type=int, default=15)
parser.add_argument('--validation_split', dest='validation_split', type=float, default=0.2)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=120)
parser.add_argument('--epochs', dest='epochs', type=int, default=250)
parser.add_argument('--temp_folder', dest='temp_folder', type=str, default='temp_folder')
args = parser.parse_args()

# assert(args.dropout <= 1.0 and args.dropout >= 0.0)
# args.dropout = True if args.dropout else False
args.dense_classifier = True if args.dense_classifier else False


from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



def embedding_trainer(G, embedder, epochs=250, seed=1234, learning_rate=0.05, embedding_dim=96, batch_size=1024, walk_length=30, num_walks=200, window=10, p=1.0, q=1.0, workers=1, temp_folder=None):
	if embedder == 'node2vec':
		if not os.path.isdir(temp_folder):
			os.mkdir(temp_folder)
		model = Node2Vec(G, dimensions=embedding_dim, 
						walk_length=walk_length, 
						num_walks=num_walks, 
						p=p,
						q=q,
						weight_key='weight',
						temp_folder=temp_folder)#, 
						#workers=workers)
		model = model.fit(window=window, min_count=1, seed=seed, alpha=learning_rate, batch_words=4)
		# embeddings = model.save()
		model.wv.save_word2vec_format('./temp_embeddings_file.emb')
		embeddings = node2vec_embedder('temp_embeddings_file.emb')
		os.remove('./temp_embeddings_file.emb')
	elif embedder == 'line':
		model = LINE(G, embedding_size=embedding_dim, order='second')
		model.train(batch_size=batch_size, epochs=epochs, verbose=2)
		embeddings = model.get_embeddings()
	return embeddings


def main(args):

	df = pd.read_csv(args.directory+'/'+args.graph_file, header=None, names=['source', 'target', 'weight'])
	G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=nx.Graph())
	# G = nx.complete_graph(100)
	full_filepath, embedd_str = make_filepath(args)
	os.mkdir(args.directory+'/'+full_filepath)

	save_embeddings = True
	if not os.path.isdir(args.directory+'/embeddings'):
		os.mkdir(args.directory+'/embeddings')

	# path.exists("guru99.txt")
	if os.path.exists(args.directory+'/embeddings/'+embedd_str+'_embedding.json'):
		save_embeddings = False
		with open(args.directory+'/embeddings/'+embedd_str+'_embedding.json', 'r') as fp:
			embeddings = json.load(fp)
		# embeddings = load(args.directory+'/'+embedd_str+'/embeddings.txt')
	else:
		embeddings = embedding_trainer(G=G, 
									embedding_dim=args.embedding_dim,
									embedder=args.embedder.lower(), 
									batch_size=args.embedding_batch_size,
									epochs=args.embedding_epochs,
									seed=args.embedding_seed,
									learning_rate=args.embedding_lr,
									walk_length=args.walk_length,
									num_walks=args.num_walks,
									window=args.window,
									p=args.p,
									q=args.q,
									workers=args.workers,
									temp_folder=args.directory+'/'+args.temp_folder)

	if save_embeddings:
		# os.mkdir(args.directory+'/embeddings/'+embedd_str+'embedding.json')
		with open(args.directory+'/embeddings/'+embedd_str+'_embedding.json', 'w') as fp:
			json.dump(embeddings, fp)
	# print(args.layers)
	# print(embeddings)
	data = Dataset(embeddings=embeddings, G=G, directory=args.directory, graph_file=args.graph_file)
	classifier = Classifier(dense_classifier=args.dense_classifier,
							embedding_dim=args.embedding_dim,
							layers=args.layers,
							dropout=args.dropout)

	train_data = data.train_data()
	test_data = data.test_data()

	filepath = args.directory+'/'+full_filepath+'/checkpoint_{epoch:02d}-{val_accuracy:.2f}.hdf5'
	classifier.train(filepath=filepath,
					patience=args.patience, 
					validation_split=args.validation_split, 
					batch_size=args.batch_size, 
					epochs=args.epochs, 
					train_data=train_data, 
					test_data=test_data)

	# get_inference_examples(self, edges_used, num_examples = 100000, attempts = 1000000):


if __name__=='__main__':
	main(args)



