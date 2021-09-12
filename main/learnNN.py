from os import path
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from lib.featureNormalize import featureNormalize
from lib.randInitializeWeights import randInitializeWeights
from lib.unpackTheta import unpackTheta

import scipy.optimize as op
from lib.nnCostFunction import nnCostFunction, fmincg_batch, fmincg_SGD
from datetime import datetime

class modelNN: 
    def __init__(self):
        pass
    def learnNN(self, X,Y, 
                nnOptions = {"Lambda":0,
                            "hidenLayers": 0,
                            "activationFn": 'sigmoid',
                            'validPercent':20,
                            'doNormalize': True,
                            'savePath': None},
                optimization = {'algorithm':'fmincg_SGD','opt_options':{"l_rate": 0.1, 'l_type': 'constant','maxIter': 500}},
                **kwargs):
        #   Input parameters
        #   X:                  The feature matrix [m,f], where f is the number of
        #                       features, and m is the number of training examples
        #   Y:                  The labels matrix [m,1]
        #   nnOptions: hyperparameters for model setting
        #          'lambda' - regularisaton parameter (numeric). The default is 0.
        #          'hidenLayers' - row vector of the number of nodes in each hidden layer. 
        #                   E.g. [20 10 5] will create a network with 3 hidden layers 
        #                   of the given number of nodes (numeric). The default is a 
        #                   single layer with the number of nodes determined as a geometric 
        #                   average of input and output layers 
        #           'nrOfLabels' - number of output labels (numeric). The default is determined as the 
        #                 number of unique values of y
        #           'activationFn' - either 'tanh' or 'sigm' (string). The default is 'sigm'
        #           'validPercent' - the percentage of examples randomly selected for validaton (numeric).
        #                   The default is 20#.
        #           'doNormalize' - 0 or 1 (numeric). If true, the data will be normalized to 0 mean and 
        #                  1 standard deviation.
        #           'savePath' - if set, the model will be saved at this location (string). The default is not set.
        
        #   Output
        #   modelNN:            The trained model. This includes confusion_train 
        #                       and confusion_valid, the confusion matrices for
        #                       training and validation sets. It also includes the
        #                       resulting model and all the parameters that go with
        #                       it.
        
        # pass in the model hyperparameters
        Lambda = nnOptions['Lambda']
        hidenLayers = nnOptions['hidenLayers']
        activationFn = nnOptions['activationFn']
        validPercent = nnOptions['validPercent']
        doNormalize = nnOptions['doNormalize']

        nrOfLabels = len(set(Y.flatten()))
        if hidenLayers==0:
            hidenLayers = round(np.sqrt(nrOfLabels*X.shape[1]))

        input_layer_size = X.shape[1]
        layers = np.array([input_layer_size, hidenLayers, nrOfLabels]) # full layer structure
        layers = layers.astype('int')
        
        self.activationFn = activationFn
        self.Lambda = Lambda
        self.layers = layers
        self.doNormalize = doNormalize
        
        # in case there are NaN elements
        X[np.isnan(X)] = 0
        # Y start from 0 (python index from 0)
        Y = Y-min(Y)
        
        # splitting out the validation set
        m = X.shape[0]
        validation_set_size = round(m*validPercent/100)
        rand_indices = np.random.permutation(m)
        X = X[rand_indices, :]
        Y = Y[rand_indices]
        X_valid = X[0:validation_set_size, :]
        Y_valid = Y[0:validation_set_size]
        X_train = X[validation_set_size:,:]
        Y_train = Y[validation_set_size:]
        
        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        
        # normalizing the features
        if doNormalize:
            X_train_norm, nnMu, nnSigma = featureNormalize(X_train)
        else:
            X_train_norm, nnMu, nnSigma  = X_train, np.repeat(0,X_train.shape[1]),np.repeat(1,X_train.shape[1])
          
        self.nnMu = nnMu
        self.nnSigma = nnSigma
          
        # initialize the weights
        initial_nn_params = randInitializeWeights(layers)
        
        # Set options for optimizer
        algorithm = optimization.get('algorithm')
        opt_options = optimization.get('opt_options')
        
        # The actual minimization happens here
        startT = datetime.now()
        
        __, theta, diag = eval(f"""{algorithm}(fun = nnCostFunction,
                                    p0 = initial_nn_params,
                                    X = X_train_norm, 
                                    Y = Y_train, 
                                    opt_options=opt_options,
                                    layers = layers, 
                                    Lambda = Lambda, 
                                    activationFn = activationFn)""")

        endT = datetime.now()
        elapsed = (endT - startT)
        print('Required CP  U Time: #f\n', elapsed)

        # unpacking the result to a cell array of weights for each layer
        self.Theta = unpackTheta(theta, layers)
        self.model_diag = diag
        
        # Computing the Prediction Accuracy
        # Predictions on the validation and training sets
        p_valid = self.predictNN(X_valid)
        p_train = self.predictNN(X_train)

        # Building confusion matrices
        confusion_train = np.zeros(nrOfLabels*nrOfLabels).reshape(nrOfLabels, nrOfLabels)
        confusion_valid = np.zeros(nrOfLabels*nrOfLabels).reshape(nrOfLabels, nrOfLabels)
        
        # rows are the actual value
        # columns are the predicted value
        for lp in np.arange(nrOfLabels):
            for la in np.arange(nrOfLabels):
                confusion_valid[la, lp] = sum((p_valid==lp)&(Y_valid==la))
                confusion_train[la, lp] = sum((p_train==lp)&(Y_train==la))
                
        # filling in model params
        self.confusion_valid = confusion_valid
        self.confusion_train = confusion_train
        self.trainingTimestamp = datetime.now()

        if (nnOptions['savePath'] is not None):
            if path.exists(nnOptions['savePath']):
                # Saving the computed parameters
                pickle_file = path.join(nnOptions['savePath'],"modelNN.pkl")
                pickle.dump(self, open(pickle_file, 'wb'))
            
    def predictNN(self, X):
        #PREDICTNN Predict the label of an input given a trained neural network
        #
        #   [p, h] = PREDICTN   N(self, X)
        #   This normalizes the data based on the mu and std in the modelNN
        #
        #
        #   Inputs
        #   self:        The trained NN model object
        #   X:              The feature matrix to run the prediction on
        #
        #   Outputs:
        #   p:              The predicted label
        #   h:              The values of the hypotheses for all labels as a vector

        Theta = self.Theta
        activationFn = self.activationFn
        # also need to normalize here
        X_mu = np.repeat(self.nnMu, X.shape[0]).reshape(X.shape[0], X.shape[1])
        X_sigma = np.repeat(self.nnSigma, X.shape[0]).reshape(X.shape[0], X.shape[1])
        X_norm = (X - X_mu)/X_sigma
        X_norm[np.isnan(X_norm)] = 0
        X_norm[np.isinf(X_norm)] = 0
 
        # choose the activation function
        if activationFn == 'tanh':
            def aF(x):
                return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) 
        else: # the default is sigmoid for now
            def aF(x):
                return 1 / (1 + np.exp(-x))

        # Number of examples
        m = X_norm.shape[0]
        
        # hypothesis values of the 1st layer   
        h = aF(np.hstack((np.ones((m,1)), X_norm))*Theta[0].T)
        # hypothesis values of the consecutive layers
        for ii in np.arange(len(Theta))[1:]:
            h = aF(np.hstack((np.ones((m,1)), h))*Theta[ii].T)

        # prediction is the highest probability value
        yhat = np.where(h == np.amax(h, 1))[1] 
        yhat = yhat.reshape(len(yhat),1)
        
        return yhat


def predictNN(self, X):
    #PREDICTNN Predict the label of an input given a trained neural network
    #
    #   [p, h] = PREDICTN   N(self, X)
    #   This normalizes the data based on the mu and std in the modelNN
    #
    #
    #   Inputs
    #   self:        The trained NN model object
    #   X:              The feature matrix to run the prediction on
    #
    #   Outputs:
    #   p:              The predicted label
    #   h:              The values of the hypotheses for all labels as a vector

    Theta = self.Theta
    activationFn = self.activationFn
    # also need to normalize here
    X_mu = np.repeat(self.nnMu, X.shape[0]).reshape(X.shape[0], X.shape[1])
    X_sigma = np.repeat(self.nnSigma, X.shape[0]).reshape(X.shape[0], X.shape[1])
    X = (X - X_mu)/X_sigma
    X[np.isnan(X)] = 0

    # choose the activation function
    if activationFn == 'tanh':
        def aF(x):
            return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) 
    else: # the default is sigmoid for now
        def aF(x):
            return 1 / (1 + np.exp(-x))

    # Number of examples
    m = X.shape[0]
    
    # hypothesis values of the 1st layer   
    h = aF(np.hstack((np.ones((m,1)), X))*Theta[0].T)
    # hypothesis values of the consecutive layers
    for ii in np.arange(len(Theta))[1:]:
        h = aF(np.hstack((np.ones((m,1)), h))*Theta[ii].T)

    # prediction is the highest probability value
    yhat = np.where(h == np.amax(h, 1))[1] 
    yhat = yhat.reshape(len(yhat),1)
    
    return yhat
        
            
            
            