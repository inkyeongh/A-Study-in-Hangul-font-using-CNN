from __future__ import print_function

import sys, os, time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import svm

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import objective
from nolearn.lasagne import PrintLayerInfo

import hangul_model
from hangul_cnn_setting import *

# down sampling
#https://github.com/aigamedev/scikit-neuralnetwork/issues/235
# Seed for reproduciblity
np.random.seed(2019)

Dataset = load_hangul_dataset(folder='../data', typ='train')
    
X_train, y_train, info_train, X_test, y_test, info_test, X_val, y_val, info_val = dat_preprocess(Dataset, test_n=20000, val_p=0.2, n_class=2)

dataset = {'train': {'X': X_train, 'y': y_train, 'info': info_train},
           'valid': {'X': X_val, 'y': y_val, 'info': info_val},
           'test': {'X': X_test, 'y': y_test, 'info': info_test}}


print("start program")

input_var = []
input_var.append( T.tensor4('inputs') )
input_var.append( T.matrix('info') )

target_var = T.ivector('targets')

network = hangul_model.model1(X_train[0].shape, info_train[0].shape, input_var)
train_history, val_history, acc_history, training_loss, validation_loss, validation_acc = main(dataset, network, input_var, target_var, 
                                                                                               batch_size = 10, num_epochs=1, 
                                                                                               filename='../model/model1_16.npz')