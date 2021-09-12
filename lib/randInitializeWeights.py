#https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init#:~:text=%20V%20a%20r%20%28W%20i%29%20%3D%201,Xavier%20%28Glorot%29%20initialization%20is%20implemented%20in%20Caffee%20library.

import numpy as np

def efun(a1,a2):
    """ function for determining the amplitude of init values for each layer
    """
    return np.sqrt(6)/np.sqrt(a1+a2) # initialise weight by uniform distribution

def randInitializeWeights(layers):
    """ RANDINITIALIZEWEIGHTS Randomly initialize the weights of a neural network
        given the structure of the layers (array of [input + hidden + output]). w is rolled back to a vector.
        eg: layers = np.array([5,5,2])
    """
    # numbers in each layer
    nel = (layers[:-1]+1)*layers[1:]
    nel = nel.astype('int')
    
    # the init apmlitudes for each layer
    epsilon_init = np.repeat(efun(layers[:-1], layers[1:]),nel)
    
    # the init weights for each neuron
    w = (2*np.random.uniform(size = sum(nel))-1)*epsilon_init
    
    return w
    
    
    
    