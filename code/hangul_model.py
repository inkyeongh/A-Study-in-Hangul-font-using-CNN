from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

# X_image & y_labels # without info
def model0(input_shape, info_shape, input_var=None):
    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, input_shape[0], input_shape[1], input_shape[2]), input_var=input_var[0])
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=4, filter_size=(5, 5),
                                              # common ReLU nonlinearity
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              # Use He et. al.'s initialization
                                              W=lasagne.init.HeNormal(gain='relu'))
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
    net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=4, filter_size=(5, 5),
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              W=lasagne.init.HeNormal(gain='relu'))
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))
    net['fc3'] = lasagne.layers.DenseLayer(net['pool2'], num_units=16, 
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.HeNormal(gain='relu'))
    net['prob'] = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(net['fc3'],p=.5),
                                            num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return net

# X_image & info & y_labels
def model1(input_shape, info_shape, input_var=None):
    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, input_shape[0], input_shape[1], input_shape[2]), input_var=input_var[0])
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=4, filter_size=(5, 5),
                                              # common ReLU nonlinearity
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              # Use He et. al.'s initialization
                                              W=lasagne.init.HeNormal(gain='relu'))
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
    net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=4, filter_size=(5, 5),
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              W=lasagne.init.HeNormal(gain='relu'))
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))

    net['fc3'] = lasagne.layers.DenseLayer(net['pool2'], num_units=16, 
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.HeNormal(gain='relu'))
    
    net['info'] = lasagne.layers.InputLayer(shape=(None, info_shape[0]), input_var=input_var[1])
    
    net['fcconcat4'] = lasagne.layers.ConcatLayer((lasagne.layers.DropoutLayer(net['fc3'],p=.5), net['info']), axis=1)
    
    net['prob'] = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(net['fcconcat4'],p=.5),
                                            num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return net
