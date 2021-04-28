# Brain-Age-Prediction-and-Age-Conditioned-Template-Generation

This repository contain the codes associated with the MIDL 2021 submission: [Unifying Brain Age Prediction and Age-Conditioned Template
Generation with a Deterministic Autoencoder](https://openreview.net/forum?id=9ClUQ2ELJap)

The repository contains 3 folders:
1. Separate training: Contains the scripts to separately train the autoencoder and the invertible module
2. Simultaneous training: Contains the scripts to simultaneously train the autoencoder and the invertible module, either from scratch or by initializing the weights using each pre-trained model.
3. Testing: Contains the scripts to test the age prediction component (see section 3.4 in the paper) and the image generation component of the model (see section 3.6 in the paper).

All the scripts contain a 'CONFIG PARAM' section, where the path to external files such as images have to be set.

The main database used in the paper is not open source, however we used the open source IXI database (<https://brain-development.org/ixi-dataset/>) to test the model, and any other similar data can be used to train and test the model. The data require preprocessing as described in the paper.

## Requirements

The code uses the common python (version 3.6.3 was used) packages numpy and pandas, as well as nibabel to read nifti images.
The Tensorflow library is used for the model development: tensorflow-gpu version 2.3.0
