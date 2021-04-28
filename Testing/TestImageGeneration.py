# author: Pauline Mouches
# This script tests the image simulation ability of the model, for different ages
#/!\ FullModel.py, SplitModel.py and TestAgePrediction.py must be run before

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

from sklearn import mixture

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

latentSpaceSize=1620

# number of images to simulate
nbSimulatedImages = 10
nbAgesToSimulate = 3 # Ex here we sample 10 times from the Gaussian mixture, and for each sampled disentangled latent space we generate the image for the age 30, 50 and 70
nbGeneratedImages = nbSimulatedImages*nbAgesToSimulate
################################################################

########################################### Fit gaussian mixture model #######################################

# You need to first use the encoder + invertible module on the training data and save the output (= the disangled latent space) to fit the gaussian mixture model
# To do so, use code from TestAgePrediction.py, with your training data as input, and save the content of 'testDisantangledLs' variable (line 88) as a npy file

ls = np.load('training_data_disentangled_ls.npy')
# Initialize the Gaussian mixture model
gmm = mixture.GaussianMixture(n_components=10,covariance_type='full')

# Keep only the component unrelated to age from the disentangled latent space (= all components except the first one)
lsWithoutAge = ls[:,1:ls.shape[1]]

# Fit the Gaussian mixture model
gmm_fitted = gmm.fit(lsWithoutAge)
print(gmm_fitted.get_params())

# Samples simulated component unrelated to age from the model
X_sampled, y_sampled = gmm_fitted.sample(nbSimulatedImages)

# Set an age to this components, here we try generate three scans per sampled simulated component, using different ages (see paper section 3.6)
X30 = np.insert(X_sampled,0,30,axis=1)
X50 = np.insert(X_sampled,0,50,axis=1)
X70 = np.insert(X_sampled,0,70,axis=1)

concat = np.concatenate((X30,X50,X70))


################################## ReconstructLatentSpace ###################################

# Load invertible module
model = tf.keras.models.load_model('invertiblemodule.hdf5',compile=False,custom_objects={'InvertibleDenseLayer': InvertibleDenseLayer})

# Built invertible module architecture
# Necessary as we want to use only one direction of the module (from disentangled latent space to latent space), so we have to split it 
input_layer = Input(shape=(latentSpaceSize,))
newDense = tf.keras.layers.Dense(units=latentSpaceSize, activation='linear',use_bias=False)
out=newDense(input_layer)
func = tf.keras.Model([input_layer],[newDense.output])

# set the weights for the backward pass (from disentangled latent space to latent space)
w = model.layers[1].get_weights()

# Transpose the weights
t = np.transpose(w[0])
wt = w
wt[0] = t
# Assign transposed weights to dense layer
newDense.set_weights(wt)

# Generate the latent spaces (from the disentangled latent spaces)
reconstructedLs = func.predict(concat)

print("done reconstruct latent space")


################################## ReconstructImage ###################################

# Load decoder
decoder = tf.keras.models.load_model('decoder.hdf5')

reconstructedLs = np.reshape(reconstructedLs,(nbGeneratedImages,latentSpaceSize))

# Use the decoder to reconstruct the simulated images from the latent spaces
reconstructedIm = decoder.predict(reconstructedLs)

# Save the images
for i in range(0,nbGeneratedImages):
	a = reconstructedIm[i,:,:,:,0]
	img = nb.Nifti1Image(a, np.eye(4))
	nb.save(img,'simulatedImage_' + str(i) + '.nii.gz')

print("done reconstruct image")