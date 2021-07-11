import sys
import os
import pandas as pd
from pathlib import Path, PurePath
CWD = os.getcwd()
sys.path.append(CWD)

import scipy.io
from learnNN import modelNN
import matplotlib.pyplot as plt
import numpy as np
from lib.plotConfMat import plotConfMat
import seaborn as sns

# Setting up
# loading example data
testData = scipy.io.loadmat('data/zero_to_nine_numbers.mat')
#testData = load('data/ini_data.mat')

# Learning
nnOptions = {"Lambda":0,
            "hidenLayers": 0,
            "activationFn": 'sigmoid',
            'validPercent':20,
            'doNormalize': True,
            'savePath': None}

#optimization = {'algorithm':'fmincg_batch','opt_options':{"l_rate": 0.1, 'l_type': 'constant','maxIter': 50}}
optimization = {'algorithm':'fmincg_SGD','opt_options':{"l_rate": 0.1, 'l_type': 'Adam','maxIter': 10, 'batch_size':50}}
m = modelNN()
m.learnNN(testData['X'], testData['y'], nnOptions=nnOptions,optimization = optimization)

# plotting the confusion matrix for the validation set
plt.figure()
plotConfMat(m.confusion_valid)

# Predicting on a random image
rI = np.random.randint(0,testData['X'].shape[0]-1) # a random index
p = m.predictNN(testData['X'][rI,:].reshape(1, testData['X'].shape[1])) # the prediction

fig, ax = plt.subplots()
ax.imshow(testData['X'][rI,:].reshape(20, 20), cmap='Greys') # set colormap to sequential Greys
title = 'Actual: {}, Predicted: {}'.format(testData['y'][rI], p)
