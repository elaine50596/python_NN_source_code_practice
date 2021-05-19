import sys
import os
import pandas as pd
sys.path
sys.path.insert(0, 'c:\\Users\\elain\\Documents\\Machine Learning\\simpleNN-Python')

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
            'opt_options': {"l_rate": 0.1, 'l_type': 'constant','ftol': 10e-6, "maxIter": 100},
            'savePath': None}

m = modelNN()
m.learnNN(testData['X'], testData['y'], nnOptions=nnOptions)

# plotting the confusion matrix for the validation set
plt.figure()
plotConfMat(m.confusion_valid)

# Predicting on a random image
rI = np.random.randint(0,testData['X'].shape[0]-1) # a random index
p = m.predictNN(testData['X'][rI,:].reshape(1, testData['X'].shape[1])) # the prediction

fig, ax = plt.subplots()
ax.imshow(testData['X'][rI,:].reshape(20, 20), cmap='Greys') # set colormap to sequential Greys
title = 'Actual: {}, Predicted: {}'.format(testData['y'][rI], p)


# compare different learning rate methods
diags = list()
methods = ['constant', 'decayed', 'scheduled','cycling','adaptive-RMSProp']
for i in np.arange(len(methods)):
    l_rate_method = methods[i]
    print(f"try learning rate method: {l_rate_method}")
    nnOptions['opt_options']['l_type'] = l_rate_method
    m = modelNN()
    m.learnNN(testData['X'], testData['y'], nnOptions=nnOptions)
    model_diag = m.model_diag
    model_diag['l_rate_method'] = l_rate_method
    diags.append(model_diag)

diags_df = pd.concat(diags,axis = 0)
sns.lineplot('iteration','objective', hue = 'l_rate_method', data = diags_df)

    



    
    
