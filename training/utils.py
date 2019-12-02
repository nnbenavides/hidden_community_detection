
def node2vec_embedder(embedding_file):
	embeddings = {}
	f = open(embedding_file, 'r')
	f.readline()
	line = f.readline()
	line = f.readline()
	while(line):
	    if line[0] == ' ': line = line[1:]
	    line = line.split(' ')
	    key = int(line[0])
	    embedding = np.array([float(x) for x in line[1:]]).astype('float')
	    embeddings[key] = embedding
	    line = f.readline()
	
	return embeddings


def make_filepath(args):
	node2vecstr = '_numwalks-%d_walklength-%d_window-%d' % (args.num_walks, args.walk_length, args.window) if args.embedder.lower()=='node2vec' else ''
	embedd_str = 'dimension-%d_lr-%.4f_seed-%d_epochs-%d%s' % (args.embedding_dim, args.embedding_lr, args.embedding_seed, args.embedding_epochs, node2vecstr)
	layer_str = ''.join([str(d)+'+' for d in args.layers])
	nn_str = ('dense_' if args.dense_classifier else 'rnn_') + ('dropout-%.2f_' % args.dropout if args.dropout is not None else '') + 'layers-%s' % layer_str[:-1]
	full_filepath = '%s-%s_NN-%s' % (args.embedder.lower(), embedd_str, nn_str)

	return full_filepath, embedd_str



arguments = ['embedding_dim', 'embedding_seed', 'embedding_lr', 'walk_length', 'num_walks', 'window', 'p', 'q']

embed_args = [['node2vec', 'line'],
		[15, 24, 32, 64, 96, 128, 256, 512, 1024],
		[1234, 4321],
		[0.001, 0.005, 0.01, 0.05, 0.0001]]

nodevecs = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
				[20, 40, 60, 80, 100, 150, 200, 300, 400], 
				[5, 10, 20, 40, 60, 80, 100], 
				[.2, .4, .6, .8, 1.0], 
				[.2, .4, .6, .8, 1.0]]

nn_args = [[False, True],[None, .25, .5], [2, 3, 4, 6, 8, 12]]
layers = [[2, 3, 4, 5, 6, 8], [1, 2, 3, 4]]

def create_arg_string():
	embedder = choice(embed_args[0])
	arg_string = '--embedder %s ' % embedder

	for i in range(1, len(embed_args)):
		arg_string += '--%s ' % arguments[i] + str(choice(embed_args[i])) + ' '

	if embedder == 'node2vec':
		for i in range(len(node2vecs)):
			arg_string += '--%s ' % arguments[i+4] + str(choice(node2vecs[i])) + ' '


	dense = choice(nn_args[0])
	dropout = choice(nn_args[1])
	layers = choice(layers[0] if dense else layers[1])

	full = gen_layers(layers)
	layer_str = ''.join([str(d)+' ' for d in layers])[:-1]
	arg_string += '--dense_classifier ' + str(1 if dense else 0) + ' '
	if dropout:
		arg_string += '--dropout %.4f ' % dropout	
	arg_string += '--layers %s' % later_str



layer1 = [32, 64, 128, 256, 512, 1024]

def gen_layers(layers, dense):
	full = []
	start = choice(layer1)
	while(dense and layers > 4 and start < 256):
		start = choice(layer1)

	for i in range(layers):
		if (not dense) and i == (layers-1):
			# if recurrent manually add the very last layer
			full.append(full[-1] if full[-1] < 32 else 32)
			break

		full.append(start)
		# randomly decide to half the next layers input unit size
		if start > 16 and np.random.rand() > .5:
			start = start/2