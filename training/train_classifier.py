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


def run_training(args):
	# df = pd.read_csv('./data/reddit_nodes_weighted_full.csv', header=None, names=['source', 'target', 'weight'])
	df = pd.read_csv(args["directory"]+'/'+args["graph_file"], header=None, names=['source', 'target', 'weight'])
	G = nx.from_pandas_edgelist(df, edge_attr='weight', create_using=nx.Graph())
	full_filepath, embedd_str = make_filepath(args)
	
	filepath = args["directory"]+'/'+full_filepath+'/checkpoint_{epoch:02d}-{val_loss:.5f}.hdf5'
	if os.path.isdir(filepath): return
	
	os.mkdir(args["directory"]+'/'+full_filepath)

	with open(args["directory"]+'/embeddings/'+args["embedding_file"], 'r') as fp:
		embeddings = json.load(fp)

	data = Dataset(embeddings=embeddings, G=G, directory=args["directory"], graph_file=args["graph_file"], embedding_dim=args["embedding_dim"])

	classifier = Classifier(dense_classifier=args["dense_classifier"],
							embedding_dim=args["embedding_dim"],
							layers=args["layers"],
							dropout=args["dropout"],
							epochs=args["epochs"],
							validation_split=args["validation_split"],
							batch_size=args["batch_size"])

	print('about to get train data')
	train_data = data.train_data()
	print('got train data')
	test_data = data.test_data()
	print('got test data')

	
	classifier.train(filepath=filepath,
					patience=args["patience"], 
					validation_split=args["validation_split"], 
					batch_size=args["batch_size"], 
					epochs=args["epochs"], 
					train_data=train_data, 
					test_data=test_data)

def main(directory='./data', 
				embedder='node2vec', 
				embedding_file='rolx_embeddings.json',
				graph_file='reddit_nodes_weighted_full.csv',
				dropout=None,
				layers=[128,64,32],
				dense_classifier=True,
				patience=10,
				validation_split=0.2,
				batch_size=120,
				epochs=1000):
	
	args = {'directory':directory,
				'embedding_file': embedding_file,
				'graph_file':graph_file,
				'dropout':dropout,
				'layers':layers,
				'dense_classifier':dense_classifier,
				'patience':patience,
				'validation_split':validation_split,
				'batch_size':batch_size,
				'epochs':epochs,
				'temp_folder':temp_folder}

	run_training(args)
# if __name__=='__main__':
	# main(args)



