# author: Pauline Mouches
# This script extracts the latent spaces from the autoencoder and saves them to train the invertible module
#/!\ AutoEncoder.py must be run before


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model

import numpy as np
import nibabel as nb

######################### CONFIG PARAM #########################

trainingIDsFile = "training.npy"#npy file containing the training participants ids
imageDirectory = "images/"

trainingIDs = np.load(trainingIDsFile)

batchSize=28

# Image size
imagex = 71
imagey = 93
imagez = 39

################################################################

# Load the encoder
model = tf.keras.models.load_model('encoderSeparateTraining.hdf5')

# Instantiate storage for the training images from which to extract the latent spaces
trainImages = np.empty((trainingIDs.shape[0], imagex,imagey,imagez, 1))

# Generate data
for i, ID in enumerate(trainingIDs):
    # Store sample
	img = nb.load(imageDirectory + str(ID) + '.nii.gz')
	a = np.array(img.dataobj)
	trainImages[i,] = a.reshape(imagex,imagey,imagez,1)

# Get the latent spaces
pred = model.predict(trainImages,batch_size=batchSize)
# Save the latent spaces
np.save('trainingLatentSpaces.npy',pred)