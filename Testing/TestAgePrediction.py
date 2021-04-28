# author: Pauline Mouches
# This script tests the age prediction ability of the model
#/!\ FullModel.py and SplitModel.py must be run before

import nibabel as nb
import tensorflow.keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Concatenate,Layer
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, UpSampling3D, BatchNormalization, Reshape, Cropping3D, ZeroPadding3D
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta,Adam
import tensorflow as tf
import numpy as np
import os
import pandas as pd


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

######################### CONFIG PARAM #########################

testID = np.load('testID.npy') #npy file containing the test participants ids
imageDirectory = "images/"
nbTestSamples = 200 #to change according to the number of samples to test

# Image size
imagex = 71
imagey = 93
imagez = 39

latentSpaceSize = 1620

batchSize=28

################################################################


################################## GetLatentSpace ###################################

testImages = np.empty((nbTestSamples, imagex,imagey,imagez, 1))

# Generate data
for i, ID in enumerate(testID):
    # Store sample
	img = nb.load(imageDirectory + str(ID) + '.nii.gz')
	a = np.array(img.dataobj)
	testImages[i,] = a.reshape(imagex,imagey,imagez,1)

# Load encoder
encoder = tf.keras.models.load_model('encoder.hdf5')

# Get the latent space for the test participant
testLs = encoder.predict(testImages,batch_size=batchSize)

print("done get latent space")

################################## EstimateAge ###################################

# Load invertible module
model = tf.keras.models.load_model('invertiblemodule.hdf5',compile=False,custom_objects={'InvertibleDenseLayer': InvertibleDenseLayer})

# Built invertible module architecture and set the weights for the forward pass (from latent space to disentangled latent space)
# Necessary as we want to use only one direction of the module (from latent space to disentangled latent space), so we have to split it 
input_layer = Input(shape=(latentSpaceSize,))
newDense = tf.keras.layers.Dense(units=latentSpaceSize, activation='linear',use_bias=False)
out=newDense(input_layer)
func = tf.keras.Model([input_layer],[newDense.output])
w = model.layers[1].get_weights()
newDense.set_weights(w)

# Get the disentangled latent space
testDisantangledLs=func.predict(testLs,batch_size=batchSize)
# save the disangled latent space, to be used in TestImageGeneration
np.save('training_data_disentangled_ls.npy',testDisantangledLs)
# Get the age (1st component of the disentangled latent space)
predAge = testDisantangledLs[:,0]

print("Predicted age:")
print(predAge)

print("done estimate age")

# Here it is also possible to modify the age and reconstruct the patient brain for a different age