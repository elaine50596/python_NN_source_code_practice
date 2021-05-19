import imagesc
import numpy as np 
import matplotlib.pyplot as plt

def plotConfMat(confmat):
    #PLOTCONFMAT plots the confusion matrix with colorscale, absolute numbers
    #   and precision normalized percentages
    #
    #   usage: 
    #   PLOTCONFMAT(confmat) plots the confmat with integers 1 to n as class labels
    #   PLOTCONFMAT(confmat, labels) plots the confmat with the specified labels
    #
    #   Arguments
    #   confmat:            a square confusion matrix
    #   labels (optional):  vector of class labels

    confmat[np.isnan(confmat)]=0 # in case there are NaN elements
    numlabels = confmat.shape[0] # number of labels

    # calculate the percentage accuracies
    confpercent = 100*confmat/(np.repeat(np.array([confmat.sum(axis = 0)]),numlabels, axis = 0))
    confpercent[np.isnan(confpercent)] = 0

    # plotting the heatmap
    fig, ax = plt.subplots(figsize = (12,12))
    ax.imshow(confpercent, cmap='Greys') # set colormap to sequential Greys
    title = 'Accuracy: {}%'.format(100*np.trace(confmat)/np.sum(confmat))
    ax.set_title(title)
    ax.set_xlabel('Target Class')
    ax.set_ylabel('Output Class')
    
    ax.set_xticks(np.arange(confpercent.shape[0]))
    ax.set_yticks(np.arange(confpercent.shape[1]))
    ax.set_xticklabels(np.arange(confpercent.shape[0]))
    ax.set_yticklabels(np.arange(confpercent.shape[1]))
    
    # Create text labels from the matrix values
    textStrings = np.array(['{}%\n{}'.format(round(confpercent.flatten(order = 'F')[i],2),confmat.flatten(order = 'F')[i]) for i in np.arange(confpercent.shape[0]*confpercent.shape[1])])
    textStrings = textStrings.reshape(confpercent.shape[0],confpercent.shape[1]).T
    
    for i in np.arange(confpercent.shape[0]):
        for j in  np.arange(confpercent.shape[0]):
            text = ax.text(j,i,textStrings[i,j], ha = 'center', va = 'center',color = 'r')
    fig.tight_layout()
    fig.show()  
    
