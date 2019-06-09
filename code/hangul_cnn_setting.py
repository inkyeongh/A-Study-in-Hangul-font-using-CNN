from __future__ import print_function

import sys, os, time

import numpy as np
import pickle
from keras.utils import np_utils
import theano
import theano.tensor as T

import lasagne

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_hangul_dataset(folder, typ, verbose=False):
    ## load hagul dataset
    with open(os.path.join(folder, '{}_data.pkl'.format(typ)), 'rb') as f:
        Dataset = pickle.load(f)
    
    return Dataset

    
def dat_preprocess(Dataset, test_n=20000, val_p=0.2, n_class=2, verbose=False):
    # chars dataset
    data_X, data_Y, data_info = Dataset['dataX'], Dataset['dataY'], Dataset['datainfo']
    # preprocessing

    # labels: sans serif=0 ==> (1,0)/ serif=1 ==> (0,1)
    #data_Y = np_utils.to_categorical(data_Y, n_class)
    data_Y = data_Y.reshape(data_Y.shape[0],)
    
    # shuffle dataset index
    np.random.seed(2019)
    idx = np.arange(data_Y.shape[0])
    np.random.shuffle(idx)

    data_X_sf    = data_X[idx,:,:,:]
    data_Y_sf    = data_Y[idx]
    data_info_sf = data_info[idx]

    val_n  = np.int((data_X_sf.shape[0]-test_n)*val_p)

    # The training and test set images and labels.
    # 101,338 train, 20,000 test
    X_train, X_test = data_X_sf[:-test_n], data_X_sf[-test_n:]
    y_train, y_test = data_Y_sf[:-test_n], data_Y_sf[-test_n:]
    info_train, info_test = data_info_sf[:-test_n], data_info_sf[-test_n:]

    # train 80%, val 20%
    X_train, X_val = X_train[:-val_n], X_train[-val_n:]
    y_train, y_val = y_train[:-val_n], y_train[-val_n:]
    info_train, info_val = info_train[:-val_n], info_train[-val_n:]
    return X_train, y_train, info_train, X_test, y_test, info_test, X_val, y_val, info_val

def iterate_minibatches(inputs, infos, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], infos[excerpt], targets[excerpt]
        
def main(dataset, network, input_var, target_var, batch_size=10, num_epochs=500, filename='model.npz'):
    train_history, val_history , acc_history = [] , [] ,[]    
    training_loss , validation_loss, validation_acc = [] , [], []
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network['prob'])
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training
    params = lasagne.layers.get_all_params(network['prob'], trainable=True)
    # Descent (SGD) with Nesterov momentum
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    # Or Use ADADELTA for updates
    updates = lasagne.updates.adadelta(loss, params)
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network['prob'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var[0], input_var[1], target_var], loss, updates=updates, on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var[0], input_var[1], target_var], [test_loss, test_acc], on_unused_input='warn')

    print("Starting training...")
    for epoch in range(num_epochs):
        print("epoch = ", epoch+1)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        
        batch_train_history = [] #train_history
        for batch in iterate_minibatches(dataset['train']['X'], dataset['train']['info'], dataset['train']['y'], batch_size, shuffle=False):
            inputs, infos, targets = batch
            train_err += train_fn(inputs, infos, targets)
            train_batches += 1
            
            batch_train_history.append(train_err)
            #if (train_batches / 100 * 100 == train_batches):
            #   print('       batch {}, err = {}'.format(train_batches, train_err))
            
        epoch_train = np.mean(batch_train_history) /train_batches
        train_history.append(epoch_train)
        
        val_err = 0
        val_acc = 0
        val_batches = 0
        batch_val_history = [] #val_history
        batch_acc_history = [] #acc_history
        for batch in iterate_minibatches(dataset['valid']['X'], dataset['valid']['info'], dataset['valid']['y'], batch_size, shuffle=False):
            inputs, infos, targets = batch
            err, acc = val_fn(inputs, infos, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            batch_val_history.append(val_err)
            batch_acc_history.append(val_acc)
        epoch_val = np.mean(batch_val_history) /val_batches
        epoch_acc = np.mean(batch_acc_history) /val_batches
        val_history.append(epoch_val)
        acc_history.append(epoch_acc)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        training_loss.append(train_err / train_batches)
        validation_loss.append(val_err / val_batches)
        validation_acc.append(val_acc / val_batches * 100)


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(dataset['test']['X'], dataset['test']['info'], dataset['test']['y'], batch_size, shuffle=False):
        inputs, infos, targets = batch
        err, acc = val_fn(inputs, infos, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
    
    np.savez(filename, *lasagne.layers.get_all_param_values(network['prob']))
    
    return train_history, val_history, acc_history, training_loss, validation_loss, validation_acc