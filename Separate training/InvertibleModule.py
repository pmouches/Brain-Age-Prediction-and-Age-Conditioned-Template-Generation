# author: Pauline Mouches
# This script trains the invertible module part of the model that perfoms latent space disentanglement 
#/!\ AutoEncoder.py and ExtractAutoEncoderLatentSpace.py must be run before

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import nibabel as nb
import numpy as np
import os
import pandas as pd

tf.compat.v1.disable_eager_execution()

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size, dim, covariates, latentSpaceMatched):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.covariates = covariates
		self.latentSpaceMatched = latentSpaceMatched
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
        # Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
        #'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))

	def __data_generation(self, list_IDs_temp):
       # Generates data containing batch_size samples
		# Initialization
		X = np.empty((self.batch_size, self.dim))
		y = np.empty((self.batch_size))
        # Generate data
		for i, ID in enumerate(list_IDs_temp):
             # Store sample
		 	lineNumber = np.where(self.latentSpaceMatched[:,0]==ID)
		 	X[i,] = self.latentSpaceMatched[lineNumber,1:latentSpaceMatched.shape[1]]
		 	y[i] = self.covariates.loc[self.covariates['Index'] == ID].AGE
		 	print(y[i])
		return X, y

class InvertibleDenseLayer(tf.keras.layers.Dense):
	#creates an invertible dense layer, which can be inverted by transposing the weights
	#the dense layer does not have bias
	def __init__(self, units, **kwargs):
		super().__init__(units, **kwargs)

	def build(self, input_shape):
		assert len(input_shape) >= 2
		input_dim = input_shape[-1]
		self.t_output_dim = input_dim
		self.kernel = self.add_weight(shape=(int(input_dim), self.units),initializer=self.kernel_initializer,name='kernel',regularizer=self.kernel_regularizer,constraint=self.kernel_constraint)
		self.built = True

	def call(self, inputs, invert=False):
		bs, input_dim = inputs.get_shape()

		kernel = self.kernel
		if invert:
			assert input_dim == self.units
			kernel = tf.keras.backend.transpose(kernel)

		output = tf.keras.backend.dot(inputs, kernel)
		output = self.activation(output)
		return output

# invertible module
def NN(inputLayer,latentSpaceSize):

	dense = InvertibleDenseLayer(units=latentSpaceSize, activation='linear',name="dense_1",use_bias=False)
	output1 = dense(inputLayer)
	output2 = dense(output1,invert=True)
	return output1,output2

		
def MSECustomLoss(inputLayer,output1,batchSize):

	def myloss(y_true,y_pred):
		predAge=tf.slice(output1,[0,0],[batchSize,1])
		#MSE between original and reconstructed latent space
		ls_loss = backend.mean(backend.square(inputLayer - y_pred))
		#MSE between true and predicted age
		age_loss = backend.mean(backend.square(y_true - predAge))
		return ls_loss+ age_loss
	return myloss

# Saves the model based on the best validation loss
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('invertiblemoduleSeparateTraining.hdf5',monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


######################### CONFIG PARAM #########################

trainingIDsFile = "training.npy"
covariatesFile = "covariates.csv" # CSV file containing one column with header "Index" containing the participants' IDs and one column with header "Age" containing the participants' age

latentSpaceSize = 1620

nbTrainingSamples = 1535
nbValidationSamples = 383

learningRate = 0.001
weightDecay = 0.007
batchSize = 128
nbEpochs = 3000

################################################################

latentSpace = np.load('trainingLatentSpaces.npy')
trainingIDs = np.load(trainingIDsFile)
covariates = pd.read_csv(covariatesFile)

# Match participant's IDs and latentSpace representations to retrieve their age based on ID in the DataGenerator
latentSpaceMatched = np.concatenate((np.reshape(trainingIDs,(nbTrainingSamples+nbValidationSamples,1)),latentSpace),axis=1)

# Split train and valid samples
IDsTrain = np.split(trainingIDs, [nbTrainingSamples],axis=0)[0]
IDsValid = np.split(trainingIDs, [nbTrainingSamples],axis=0)[1]
#Generate training and validation batches
train_generator = DataGenerator(IDsTrain, batchSize, (latentSpaceSize), covariates, latentSpaceMatched)
valid_generator = DataGenerator(IDsValid, batchSize, (latentSpaceSize), covariates, latentSpaceMatched)

#Build model
inputLayer = Input(shape=(latentSpaceSize,), name="Input")
output1, output2 = NN(inputLayer,latentSpaceSize)
model = Model([inputLayer], [output2])
print(model.summary())

#Train model
model.compile(loss=MSECustomLoss(inputLayer,output1,batchSize),optimizer=Adam(lr=learningRate, decay=weightDecay))
history = model.fit(train_generator, epochs=nbEpochs, validation_data=valid_generator, callbacks=[checkpoint_callback])