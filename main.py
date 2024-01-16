#!/usr/bin/env python
# coding: utf-8

# # Computing on the Dynex Neuromorphic Platform: Image Classification

# Computing on Quantum or neuromorphic systems is fundamentally different than using traditional hardware and is a very active area of research with new algorithms surfacing almost on a weekly basis. In this article we will use the Dynex SDK (beta) to perform an image classification task using a transfer learning approach based on a Quantum-Restricted-Boltzmann-Machine ("QRBM") based on the paper "A hybrid quantum-classical approach for inference on restricted Boltzmann machines". It is a step-by-step guide on how to utilize neuromorphic computing with Python using Tensorflow. This example is just one of multiple possibilities to perform machine learning tasks. However, it can be easily adopted to other use cases.

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.linear_model import LogisticRegression
# The Dynex Platform can be used as a Neuromorphic Tensorflow layer. We import the required classes:

from HybridQRBM.TensorflowQRBM import QRBM
from HybridQRBM.optimizers import RBMOptimizer
from HybridQRBM.samplers import DynexSampler
from HybridQRBM.samplers import PersistentContrastiveDivergenceSampler

# ## Parameters

# We define the training hyperparameters. Neuromorphic Dynex layers evolve extremely fast towards an nearly optimal ground state. We therefore only need a few training epochs for a fully trained model.

INIT_LR    = 1e-3  # initial loss rate for optimizer
BATCH_SIZE = 10000 # number of images per batch
EPOCHS     = 1     # number of training epochs
device = "cpu"     # no GPU needed, we compute on the Dynex Platform

optimizer = RBMOptimizer(
                learning_rate=0.05,
                momentum=0.9,
                decay_factor=1.00005,
                regularizers=()
            );

sampler = DynexSampler(mainnet=True, 
               num_reads=100000, 
               annealing_time=200,  
               debugging=False, 
               logging=True, 
               num_gibbs_updates=1, 
               minimum_stepsize=0.002);

#sampler = PersistentContrastiveDivergenceSampler(300)

# We define a model with just one Dynex neuromorphic layer. This layer is designed to find energy ground states for the entire batch of images - all these are computed fully parallel on the Dynex Platform. Technically, it is a QRBM (Quantum Restricted Boltzmann Machine) Layer returning hidden node weights. To classify we will apply a simple logistic regression model based on the hidden layers of this layer.


# ## Load MNIST Dataset

#####################################

class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.
    
    def __call__(self, x, labels):
        # Assuming the transformation is to be applied on features
        x = tf.cast(x > self.thr, x.dtype)
        return x, labels

def load_and_transform_dataset(batch_size, transform):
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    # Select half of the training data
    #half_train_images = train_images[:len(train_images) // 2]
    #half_train_labels = train_labels[:len(train_labels) // 2]

    # Select half of the testing data
    #half_test_images = test_images[:len(test_images) // 2]
    #half_test_labels = test_labels[:len(test_labels) // 2]

    #x_train = x_train[0:30000,:,:]
    #y_train = y_train[0:30000]
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Apply the custom transformation
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.map(transform).batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.map(transform).batch(batch_size)

    return train_data, test_data


# Initialize the data transformer
data_transformer = ThresholdTransform(thr_255=128)

# Load and transform the datasets
trainDataLoader, testDataLoader = load_and_transform_dataset(BATCH_SIZE, data_transformer)

print("[INFO] MNIST dataset loaded")

#############################################

# ## Model Training

# The model accumulates all 60,000 train images and processes the sampling in parallel for the entire epoch:

steps_per_epoch = len(trainDataLoader) # // BATCH_SIZE
n_hidden = 300



class QRBM_Model(tf.keras.Model):
  def __init__(self, n_hidden, steps_per_epoch, sampler, optimizer):
    
    super(QRBM_Model, self).__init__()
    self.QRBMlayer = QRBM(n_hidden, steps_per_epoch, sampler, optimizer)

  def call(self, x):
    x = self.QRBMlayer(x) 
    return x

  def save(self, path):
    pass

  def load(self, path, file):
    pass    

model = QRBM_Model(n_hidden, steps_per_epoch, sampler, optimizer)

print("[INFO] Model initialized...")

for e in range(1, EPOCHS+1):
    print('EPOCH',e,'of',EPOCHS);
    # loop over the training set
    for i, (x, y) in enumerate(trainDataLoader):
        # perform a forward pass and calculate the training loss
        pred = model(x);
    
print('FOUND MODEL ACCURACY:',np.array(model.QRBMlayer.acc).max(),'%')

## save the trained model
model.QRBMlayer.save_model("./model/QRBM.model")

# visualize progress:
plt.figure()
plt.plot(model.QRBMlayer.errors, label='QRBM_Model')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
print(model.QRBMlayer.errors)


# ## visualize test dataset:

num_samp = 0;
num_batches = 0;
for batch_idx, (inputs, targets) in enumerate(testDataLoader):
    num_samp += len(inputs);
    num_batches += 1;
    
print(num_batches,' batches total, ', num_samp,' images in total, one batch ',len(inputs),' images')

# we use data from the last batch:
fig = plt.figure(figsize=(10, 7));
fig.suptitle('Test Dataset (50 samples)', fontsize=16)
rows = 5;
columns = 10;

for j in range(0,50):
    fig.add_subplot(rows, columns, j+1)
    plt.imshow(inputs[j,:,:])
    marker=str(targets[j].numpy())
    plt.title(marker)
    plt.axis('off');
plt.show();


# ## Transfer Learning: Classifyer with Logistic Regression from RBM's Hidden Layer 

data = [];
data_labels = [];
error = 0;
for i in range(0, 50):
    inp = np.array(inputs[i,:,:].numpy().flatten()); 
    tar = np.array(targets[i])
    data.append(inp)
    data_labels.append(tar)
data = np.array(data)
data_labels = np.array(data_labels)

# extract hidden layers from RBM:
hidden, prob_hidden = model.QRBMlayer.sampler.infer(data)

# Logistic Regression classifier on hidden nodes:
t = hidden * prob_hidden
clf = LogisticRegression(max_iter=10000)
clf.fit(t, data_labels)
predictions = clf.predict(t)
print('Accuracy:', (sum(predictions == data_labels) / data_labels.shape[0]) * 100,'%')


# inspect predictions:
print('target   :',data_labels[:30])
print('predicted:',predictions[:30])


# ## Image reconstruction with the RBM


_, features = model.QRBMlayer.sampler.predict(data, num_particles=10, num_gibbs_updates=1)


fig = plt.figure(figsize=(10, 7));
fig.suptitle('Reconstructed Dataset (50 samples)', fontsize=16)
rows = 5;
columns = 10;
for i in range(0,50):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(features[i].reshape(28,28))
    marker=str(predictions[i])+' (t='+str(data_labels[i])+')'
    plt.title(marker)
    plt.axis('off');
plt.show()


# ## Load Model and Predict
model.QRBMlayer.load_model("./model/QRBM.model")

model.QRBMlayer.num_visible = model.QRBMlayer.biases_visible.shape[0] 
model.QRBMlayer.num_hidden = model.QRBMlayer.biases_hidden.shape[0] 
_, features = model.QRBMlayer.sampler.predict(data, num_particles=10,num_gibbs_updates=1)
# extract hidden layers from RBM:
hidden, prob_hidden = model.QRBMlayer.sampler.infer(data)
# Logistic Regression classifier on hidden nodes:
        
t = hidden * prob_hidden
clf = LogisticRegression(max_iter=10000)
clf.fit(t, data_labels)
predictions = clf.predict(t)
print('Accuracy:', (sum(predictions == data_labels) / data_labels.shape[0]) * 100,'%')
    
fig = plt.figure(figsize=(10, 7));
fig.suptitle('Reconstructed Dataset (50 samples)', fontsize=16)
rows = 5;
columns = 10;
for i in range(0,50):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(features[i].reshape(28,28))
    marker=str(predictions[i])+' (t='+str(data_labels[i])+')'
    plt.title(marker)
    plt.axis('off');
plt.show()