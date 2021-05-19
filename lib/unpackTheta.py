import numpy as np

def unpackTheta(theta, layers):
    #UNPACKTHETA given a vector theta and a vector of layer structure
    #   unpacks theta into a list of arrays Theta[0], Theta[1], etc.. for each layer and 

    # number of layers, excluding input
    nl = len(layers)-1
    # to store unpacked thetas
    Theta = [[] for i in range(2)]
    
    # number in each layer
    nel = (layers[:-1]+1)*layers[1:]
    
    # unpacking theta
    for ii in range(nl):
        Theta[ii] = np.mat(theta[range(nel[ii])].reshape(layers[ii+1], layers[ii]+1))
        theta = theta[nel[ii]:]
        
    return Theta
        