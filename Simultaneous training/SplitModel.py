# author: Pauline Mouches
# This script splits the full model into the encoder, the decoder and the invertible module, to be able to use them separately for testing
#/!\ FullModel.py must be run before

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Concatenate,Layer
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, UpSampling3D, BatchNormalization, Reshape, Cropping3D, ZeroPadding3D
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta,Adam

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


def decoder(latentSpace):

	# Reshape the flatten laten space to the econvBottleneck111Ventricles output shape
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

######################### CONFIG PARAM #########################

savedModel = "savedModel.hdf5"
covariatesFile = "covariates.csv" # CSV file containing one column with header "Index" containing the participants' IDs and one column with header "Age" containing the participants' age

latentSpaceSize = 1620

################################################################
cov = pd.read_csv(covariatesFile)
IDs = cov['Index'].to_numpy()


# Load the saved model
m = tf.keras.models.load_model('savedModel.hdf5',custom_objects={'InvertibleDenseLayer':InvertibleDenseLayer},compile=False)
print(m.summary())
# Splits the model to get the encoder
encoder = Model(m.layers[0].input,m.layers[15].output)

tf.keras.models.save_model(encoder,'encoder.hdf5')

# Splits the model to get the decoder
# Recreate a model with the decoder architecture
inp = Input(shape=(latentSpaceSize), name="Input")
outp = decoder(inp)
decoder = Model(inp,outp)
# Set the model weights using the weights from the saved model
for i in range(1,16): # starts from 1 as 0 is the input layer
	decoder.layers[i].set_weights(m.layers[i+16].get_weights())
# Save the decoder
tf.keras.models.save_model(decoder,'decoder.hdf5')

# Splits the model to get the invertible module
# Recreate a model with the invertible module architecture
inp1=Input((latentSpaceSize))
dense = InvertibleDenseLayer(units=latentSpaceSize, activation='linear',name="dense_1")
output1 = dense(inp1)
output2 = dense(output1,invert=True)
invertibleModule = Model(inp1,output2)
# Set the model weights using the weights from the saved model
invertibleModule.layers[1].set_weights(m.layers[16].get_weights())
# Save the invertible module
tf.keras.models.save_model(invertibleModule,'invertiblemodule.hdf5')