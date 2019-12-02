import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class Classifier:
	def __init__(self, dense_classifier, embedding_dim, layers, dropout):
		self.model = self.build_dense_classifier(embedding_dim, layers, dropout) if dense_classifier else self.build_recurrent_classifier(embedding_dim, layers, dropout)
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split

	def build_dense_classifier(self, embedding_dim, layers, dropout):
		input1 = Input(shape=(embedding_dim*2,))

		h1 = Dense(layers[0], activation='relu')(input1)
		if dropout is not None:
			h1 = Dropout(rate=dropout)(h1)

		for layer in layers[1:]:
			h1 = Dense(layer, activation='relu')(h1)
			if dropout is not None:
				h1 = Dropout(rate=dropout)(h1)

		out = Dense(2, activation='softmax')(h1)
		model = Model(inputs=input1, outputs=out)

		return model


	def build_recurrent_classifier(self, embedding_dim, layers, dropout):
		input1 = Input(shape=(embedding_dim*2,))

		h1 = LSTM(layers[0], activation='tanh')(input1)
		if dropout is not None:
			h1 = Dropout(rate=dropout)(h1)

		for layer in layers[1:]:
			h1 = LSTM(layer, activation='tanh')(h1)
			if dropout is not None:
				h1 = Dropout(rate=dropout)(h1)

		h1 = Dense(layers[-1], activation='relu')(h1)
		out = Dense(2, activation='softmax')(h1)
		model = Model(inputs=input1, outputs=out)

		return model



	def train(self, filepath, patience=10, validation_split=.2, batch_size=120, epochs=250, train_data=None, test_data=None):

		train_embeddings, train_labels = train_data
		checkpointing = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
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

		loss, acc = self.model.evaluate(x=test_embeddings, y=test_labels)

		print("Best model has test accuracy: %.4f and test loss %.4f" % (loss, acc))