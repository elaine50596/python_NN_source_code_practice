import numpy as np

def efun(a1,a2):
    """ function for determining the amplitude of init values for each layer
    """
    return np.sqrt(6)/np.sqrt(a1+a2)

def randInitializeWeights(layers):
    """ RANDINITIALIZEWEIGHTS Randomly initialize the weights of a neural network
        given the structure of the layers. w is rolled back to a vector.
    """
    # numbers in each layer
    nel = (layers[:-1]+1)*layers[1:]
    nel = nel.astype('int')
    
    # the init apmlitudes for each layer
    epsilon_init = np.repeat(efun(layers[:-1], layers[1:]),nel)
    
    # the init weights for each neuron
    w = (2*np.random.uniform(size = sum(nel))-1)*epsilon_init
    
    return w
    
    
    
    