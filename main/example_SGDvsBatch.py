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

# compare different optimization algorithm: batch vs. SGD
# for the same number of epoches/iterations, SGD takes longer time as compared to batch but converges faster
# reduce the number of epoches for SGD algorithm or increase batch size
diags = list()
optimization_batch = {'algorithm':'fmincg_batch','opt_options':{"l_rate": 0.1, 'l_type': 'constant','maxIter': 50}}
m1 = modelNN()
m1.learnNN(testData['X'], testData['y'], nnOptions=nnOptions,optimization = optimization_batch)
model_diag1 = m1.model_diag
model_diag1['algorithm'] = 'batch'
diags.append(model_diag1)

batch_size = testData['X'].shape[0]//8
optimization_SGD = {'algorithm':'fmincg_SGD','opt_options':{"l_rate": 0.1, 'l_type': 'constant','maxIter': 10, 'batch_size': batch_size}}
m2 = modelNN()
m2.learnNN(testData['X'], testData['y'], nnOptions=nnOptions,optimization = optimization_SGD)
model_diag2 = m2.model_diag
model_diag2['algorithm'] = 'SGD'
diags.append(model_diag2)

diags_df = pd.concat(diags,axis = 0)
sns.lineplot('iteration','objective', hue = 'algorithm', data = diags_df)


    
    
