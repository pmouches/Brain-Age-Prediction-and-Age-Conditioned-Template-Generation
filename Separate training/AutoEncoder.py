# author: Pauline Mouches
# This script trains the autoencoder part of the model 


import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, UpSampling3D,Cropping3D, ZeroPadding3D
from tensorflow.keras.layers import Dropout, Input, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd 
import nibabel as nb

tf.compat.v1.disable_eager_execution()

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size, dim, imageDirectory):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.imageDirectory = imageDirectory
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
        #Updates indexes after each epoch
		self.indexes = np.arange(len(self.list_IDs))

	def __data_generation(self, list_IDs_temp):
       # Generates data containing batch_size samples
        # Initialization
		X = np.empty((self.batch_size, *self.dim, 1))
		y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
		for i, ID in enumerate(list_IDs_temp):
            # Store sample
			img = nb.load(self.imageDirectory + str(ID) + '.nii.gz')
			a = np.array(img.dataobj)

			X[i,] = a.reshape(*self.dim,1)
			y[i,] = a.reshape(*self.dim,1)
		return X, y

def encoder(inputLayer):

	# Because of 3 maxpool (2*2*2) input size must be padded
	pad = ZeroPadding3D(padding=((0,1),(0,3),(0,1)),name="epadVentricles")(inputLayer)

	## Branch 1 layer name = #Branch#Block#layer
	# convolutional layers Block 1
	encoder=Conv3D(filters=8, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econv111Ventricles")(pad)
	encoder=Conv3D(filters=8, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econv112Ventricles")(encoder)
	encoder=BatchNormalization(name="enorm111Ventricles")(encoder)
	encoder=MaxPool3D(pool_size=(2, 2, 2),padding='same',name="emaxpool111Ventricles")(encoder)

	# convolutional layers Block 2
	encoder=Conv3D(filters=16, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econv121Ventricles")(encoder)
	encoder=Conv3D(filters=16, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econv122Ventricles")(encoder)
	encoder=BatchNormalization(name="enorm121Ventricles")(encoder)
	encoder=MaxPool3D(pool_size=(2, 2, 2),padding='same',name="emaxpool121Ventricles")(encoder)

	# convolutional layers Block 3
	encoder=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econv131Ventricles")(encoder)
	encoder=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econv132Ventricles")(encoder)
	encoder=BatchNormalization(name="enorm131Ventricles")(encoder)
	encoder=MaxPool3D(pool_size=(2, 2, 2),padding='same',name="emaxpool131Ventricles")(encoder)

	# Last conv reducing latent space
	encoder = Conv3D(filters=3, kernel_size=(3, 3, 3),padding='same',activation="relu",name="econvBottleneck111Ventricles")(encoder)

	latentSpace=Flatten(name="flat111Ventricles")(encoder)

	return latentSpace

def decoder(latentSpace):
	decoder=Reshape((9,12,5,3),name="reshape111Ventricles")(latentSpace)

	# first conv on reduced latent space
	decoder= Conv3D(filters=3, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconvBottleneck111Ventricles")(decoder)

	# de convolutional layers Block 3
	decoder=UpSampling3D(size=(2, 2, 2),name="dupsample111Ventricles")(decoder)
	decoder=BatchNormalization(name="dnorm111Ventricles")(decoder)
	decoder = Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconv111Ventricles")(decoder)
	decoder = Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconv112Ventricles")(decoder)

	# de convolutional layers Block 2
	decoder=UpSampling3D(size=(2, 2, 2),name="dupsample121Ventricles")(decoder)
	decoder=BatchNormalization(name="dnorm121Ventricles")(decoder)
	decoder = Conv3D(filters=16, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconv121Ventricles")(decoder)
	decoder = Conv3D(filters=16, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconv122Ventricles")(decoder)

	# de convolutional layers Block 1
	decoder=UpSampling3D(size=(2, 2, 2),name="dupsample131Ventricles")(decoder)
	decoder=BatchNormalization(name="dnorm131Ventricles")(decoder)
	decoder = Conv3D(filters=8, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconv131Ventricles")(decoder)
	decoder = Conv3D(filters=8, kernel_size=(3, 3, 3),padding='same',activation="relu",name="dconv132Ventricles")(decoder)

	# output layer
	outputLayerAE = Conv3D(filters=1, kernel_size=(3, 3, 3),padding='same',activation="linear",name="dconvz")(decoder)

	# Because input size was padded, must crop
	outputLayerAE = Cropping3D(cropping=((0,1),(0,3),(0,1)),name="outputLayerAE")(outputLayerAE)
	return outputLayerAE

# Saves the model based on the best validation loss
checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint('autoencoderSeparateTraining.hdf5',monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


######################### CONFIG PARAM #########################

imageDirectory = "images/"
trainingIDsFile = "training.npy"
savedModel = "savedModel.hdf5"
covariatesFile = "covariates.csv" # CSV file containing one column with header "Index" containing the participants' IDs and one column with header "Age" containing the participants' age

latentSpaceSize = 1620

nbTrainingSamples = 1535
nbValidationSamples = 383
nbTestingSamples = 200

imagex = 71
imagey = 93
imagez = 39

batchSize = 28
learningRate = 0.005
weightDecay = 0.007
nbEpochs = 100

################################################################

# Extract covariates
cov = pd.read_csv(covariatesFile)
trainingIDs = np.load(trainingIDsFile)
IDs = cov['Index'].to_numpy()

# Split train and valid samples
IDsTrain = np.split(trainingIDs, [nbTrainingSamples],axis=0)[0]
IDsValid = np.split(trainingIDs, [nbTrainingSamples],axis=0)[1]

# Generate input batches
training_generator = DataGenerator(IDsTrain, batchSize, (imagex,imagey,imagez), imageDirectory)
valid_generator = DataGenerator(IDsValid, batchSize, (imagex,imagey,imagez), imageDirectory)


# Built the model
inputLayerAE = Input(shape=(imagex,imagey,imagez,1), name="InputVentricles")
latentSpace = encoder(inputLayerAE)
outputLayerAE = decoder(latentSpace)

#Compile and fit the model
model = Model([inputLayerAE], [outputLayerAE])
model.compile(loss='mean_squared_error', optimizer=Adam(lr=learningRate, decay=weightDecay), metrics=['mse'])
history = model.fit(training_generator, epochs=nbEpochs, validation_data=valid_generator, callbacks=[checkpoint_callback])
#400
# Splits the model to get the encoder
encoder = Model(model.layers[0].input,model.layers[15].output)

tf.keras.models.save_model(encoder,'encoderSeparateTraining.hdf5')
