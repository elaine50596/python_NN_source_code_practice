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

# compare different basic learning rate methods on batch algorithm
diags = list()
optimization = {'algorithm':'fmincg_batch','opt_options':{"l_rate": 0.1, 'l_type': 'constant','maxIter': 50}}
methods = ['constant', 'decayed', 'scheduled','cycling','adaptive-RMSProp']
for i in np.arange(len(methods)):
    l_rate_method = methods[i]
    print(f"try learning rate method: {l_rate_method}")
    optimization['opt_options']['l_type'] = l_rate_method
    m = modelNN()
    m.learnNN(testData['X'], testData['y'], nnOptions=nnOptions, optimization=optimization)
    model_diag = m.model_diag
    model_diag['l_rate_method'] = l_rate_method
    diags.append(model_diag)

diags_df = pd.concat(diags,axis = 0)
sns.lineplot('iteration','objective', hue = 'l_rate_method', data = diags_df)

# compare different adaptive learning rate methods on SGD algorithm
diags = list()
optimization = {'algorithm':'fmincg_SGD','opt_options':{"l_rate": 0.1, 'l_type': 'adaptive-RMSProp','maxIter': 10, 'batch_size':50}}
methods = ['adaptive-RMSProp', 'Adam', 'Momentum','AdaGrad']
for i in np.arange(len(methods)):
    l_rate_method = methods[i]
    print(f"try learning rate method: {l_rate_method}")
    optimization['opt_options']['l_type'] = l_rate_method
    m = modelNN()
    m.learnNN(testData['X'], testData['y'], nnOptions=nnOptions, optimization=optimization)
    model_diag = m.model_diag
    model_diag['l_rate_method'] = l_rate_method
    diags.append(model_diag)

diags_df = pd.concat(diags,axis = 0)
sns.lineplot('iteration','objective', hue = 'l_rate_method', data = diags_df)


