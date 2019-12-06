import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape
from tensorflow.keras import Model, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class Classifier:
	def __init__(self, dense_classifier, embedding_dim, layers, dropout, batch_size, epochs, validation_split):
		self.model = self.build_dense_classifier(embedding_dim, layers, dropout) if dense_classifier else self.build_recurrent_classifier(embedding_dim, layers, dropout)
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split
		self.model.compile(optimizer='adam', loss=MeanSquaredError())

	def build_dense_classifier(self, embedding_dim, layers, dropout):
		input1 = Input(shape=(embedding_dim*2,))

		h1 = Dense(layers[0], activation='relu')(input1)
		if dropout is not None:
			h1 = Dropout(rate=dropout)(h1)

		for layer in layers[1:]:
			h1 = Dense(layer, activation='relu')(h1)
			if dropout is not None:
				h1 = Dropout(rate=dropout)(h1)

		out = Dense(1)(h1)
		model = Model(inputs=input1, outputs=out)

		return model


	def build_recurrent_classifier(self, embedding_dim, layers, dropout):
		input1 = Input(shape=(embedding_dim*2,))

		input1_reshape = Reshape((1, embedding_dim*2))(input1)
		if len(layers) == 2:
			h1 = LSTM(layers[0])(input1_reshape)
		else:
			h1 = LSTM(layers[0], return_sequences=True)(input1_reshape)
			for i in range(1, len(layers)-2):
				h1 = LSTM(layers[i], return_sequences=True)(h1)
			h1 = LSTM(layers[-2])(h1)


		h1 = Dense(layers[-1], activation='relu')(h1)
		out = Dense(1)(h1)
		model = Model(inputs=input1, outputs=out)

		return model



	def train(self, filepath, patience=10, validation_split=.2, batch_size=120, epochs=250, train_data=None, test_data=None, use_gpu=True):
		print('here at train')
		train_embeddings, train_labels = train_data
		checkpointing = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
		halting = EarlyStopping(monitor='val_loss', patience=patience, mode='auto', restore_best_weights=True)
		callbacks = [checkpointing, halting]
		self.model.fit(x=train_embeddings, y=train_labels,
					batch_size=self.batch_size,
					epochs=self.epochs,
					verbose=1,
					validation_split=self.validation_split,
					shuffle=True,
					callbacks=callbacks)

		test_embeddings, test_labels = test_data

		loss = self.model.evaluate(x=test_embeddings, y=test_labels)

		print("Best model has test accuracy: %.4f and test loss %.4f" % loss)