# author: Pauline Mouches
# This script trains the full model
#/!\ For optimized training, run the scripts from the directory 'Separate training' first
#/!\ Once the separate models are trained, their weights will be used to initialize the full model weights 

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, UpSampling3D,Cropping3D, ZeroPadding3D, Input, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend

import numpy as np
from numpy import genfromtxt
import pandas as pd 
import nibabel as nb
import os

import csv
import sys

tf.compat.v1.disable_eager_execution()

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size, dim, imageDirectory,covariates):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.imageDirectory = imageDirectory
		self.covariates = covariates		
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
       # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
		X = np.empty((self.batch_size, *self.dim, 1))
		# For any reason, tf complains when size(y) != size(output layer)
		# As we want to store the true age in y, we set size(y) == size(output layer)
		# And fill y with the true age value (line 68)
		y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
		for i, ID in enumerate(list_IDs_temp):
            # Store sample
			img = nb.load(self.imageDirectory + str(ID) + '.nii.gz')
			a = np.array(img.dataobj)
			X[i,] = a.reshape(*self.dim,1)
			y[i,].fill(int(self.covariates.loc[self.covariates['Index'] == ID].AGE))
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

# invertible module
def NN(inputLayer,latentSpaceSize,trainOption):

	dense = InvertibleDenseLayer(units=latentSpaceSize, kernel_initializer=tf.keras.initializers.Identity(), activation='linear', name="dense_1", trainable=trainOption)
	#output1 = disentangled latent space
	output1 = dense(inputLayer)
	#output2 = reconstructed latent space
	output2 = dense(output1,invert=True)
	return output1,output2

def MSECustomLoss(inputLayer,output1,output2,latentSpace,batchSize):

	def myloss(y_true,y_pred):
		#extract the age component from the disentangled latent space
		predAge = tf.slice(output1,[0,0],[batchSize,1])
		#extract the true age from y_true (=output of the DataGenerator)
		extractedAge = y_true[:,0,0,0,0]
		trueAge = backend.reshape(extractedAge,shape=(batchSize,1))
		#MSE between original and reconstructed latent space
		ls_loss = backend.mean(backend.square(latentSpace - output2))
		#MSE between true and predicted age
		age_loss = backend.mean(backend.square(trueAge - predAge))
		#Lae loss: MSE between original and reconstructed image
		AE_loss = backend.mean(backend.square(backend.flatten(inputLayer) - backend.flatten(y_pred)))
		loss = AE_loss + 158*(ls_loss+age_loss)
		return loss
	return myloss

checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint('savedModel.hdf5',monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


######################### CONFIG PARAM #########################

imageDirectory = "images/"
preTrainedAE = "autoencoderSeparateTraining.hdf5" # If separate training was performed first
preTrainedInvertibleModule = "invertiblemoduleSeparateTraining.hdf5" # If separate training was performed first
trainingIDsFile = "training.npy"
covariatesFile = "covariates.csv" # CSV file containing one column with header "Index" containing the participants' IDs and one column with header "AGE" containing the participants' age

latentSpaceSize = 1620

nbTrainingSamples = 1535
nbValidationSamples = 383

imagex = 71
imagey = 93
imagez = 39

learningRate = 0.0001
weightDecay = 0.007
batchSize = 28
nbEpochs = 100

################################################################
cov = pd.read_csv(covariatesFile)
IDs = cov['Index'].to_numpy()
trainingIDs = np.load(trainingIDsFile)

# Split train and valid samples
IDsTrain = np.split(trainingIDs, [nbTrainingSamples],axis=0)[0]
IDsValid = np.split(trainingIDs, [nbTrainingSamples],axis=0)[1]

# Generate input batches
training_generator = DataGenerator(IDsTrain, batchSize, (imagex,imagey,imagez), imageDirectory, cov)
valid_generator = DataGenerator(IDsValid, batchSize, (imagex,imagey,imagez), imageDirectory, cov)

# Built the model
inputLayerAE = Input(shape=(imagex,imagey,imagez,1), name="InputVentricles")
latentSpace = encoder(inputLayerAE)
output1, output2 = NN(latentSpace,latentSpaceSize,True)
outputLayerAE = decoder(output2)
model = Model([inputLayerAE], [outputLayerAE])

print(model.summary())

# If separate training was performed first: load weights from independently trained models
model.load_weights(preTrainedAE,by_name=True)
model.load_weights(preTrainedInvertibleModule,by_name=True)

#Compile and fit the model
model.compile(loss=MSECustomLoss(inputLayerAE,output1,output2,latentSpace,batchSize),optimizer=Adam(lr=learningRate, decay=weightDecay))
history = model.fit(training_generator, epochs=nbEpochs, validation_data=valid_generator, callbacks=[checkpoint_callback])
