import numpy as np
import pandas as pd 
import random

def unpackTheta(theta, layers):
    #UNPACKTHETA given a vector theta and a vector of layer structure
    #   unpacks theta into a list of arrays Theta[0], Theta[1], etc.. for each layer and 

    # number of layers, excluding input
    nl = len(layers)-1
    # to store unpacked thetas
    Theta = [[] for i in range(2)]
    
    # number in each layer
    nel = (layers[:-1]+1)*layers[1:]
    nel = nel.astype('int')
    
    # unpacking theta
    for ii in range(nl):
        Theta[ii] = np.mat(theta[range(nel[ii])].reshape(layers[ii+1], layers[ii]+1))
        theta = theta[nel[ii]:]
        
    return Theta

def nnCostFunction(nn_params, layers, X, Y, Lambda, activationFn):
    #NNCOSTFUNCTION Implements the neural network cost function for a multi-layer
    #   neural network which performs classification. It accepts 'sigm' or
    #   'tanh' as activation functions. Computes the cost and gradient of the NN.
    #   
    #   usage: J, grad = NNCOSTFUNCTION(nn_params, layers, X, y, Lambda) 
    #
    #   Input
    #   nn_params:      current weights of the network, collected into a single
    #                   vector
    #   layers:         the structure of the NN, including the input layer and
    #                   the output layer. E.g. [400 80 30 20 10]
    #   X:              the feature matrix
    #   y:              the labels (answers)
    #   Lambda:         regluarisation parameter
    #   activationFn:   'tanh' or 'sigm'
    #
    #   Returns
    #   J:              the value of the cost function
    #   grad:           the gradients for each neuron

    # unpacking into a list of arrays (weights)
    Theta = unpackTheta(nn_params, layers)

    # Number of observations
    m = X.shape[0]
    # number of layers excluding the output
    nl = len(layers) - 1
    # number of labels 
    num_labels = layers[-1]
    
    # number in each layer (useful for filling in gradients)
    nel = (layers[:-1]+1)*layers[1:]
    snel = np.insert(nel, 0, 0).cumsum() # indices for boundaries
    snel[1:] = np.array(snel[1:])-1 # python indices start from zero
    
    # Choose the activation function
    if activationFn == 'tanh':
        def aF(x):
            return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        def gF(x):
            return 1-aF(x)**2
        # creating the array to hold vectorized y (flat for now; reshape later)
        y_nn_flat = -np.ones(m*num_labels)    
    else: # the default is sigmoid for now
        def aF(x):
            return 1 / (1 + np.exp(-x))
        def gF(x):
            return aF(x)*(1 - aF(x))
        # creating the array to hold vectorized y (flat for now; reshape later)
        y_nn_flat = np.zeros(m*num_labels)
    
    # converting y to the binary matrix of answers (use numpy ravel_multi_index function)
    idx = np.ravel_multi_index(np.array([np.arange(m),Y.flatten()]),[m,num_labels])   
    y_nn_flat[idx]=1
    y_nn = y_nn_flat.reshape((m,num_labels))  # (y: actual) 

    # Forward Propogation---------------------
    a_t = X # the starting vals
    a = [[] for i in range(nl)] # a: after activation (output) for the current layer, input for the next layer
    z = [[] for i in range(nl+1)] # z: before activation (input) for the current layer, output from the previous layer
    for ii in range(nl):
        a[ii] = np.hstack((np.ones((m,1)), a_t)) # adding the bias column
        z[ii+1] = a[ii]*Theta[ii].T
        a_t = aF(z[ii+1])
    # hypotheses values (y-hat: predicted)
    yhat_nn = a_t

    # computing the cost (cost function varies for different problems & activation functions)
    if activationFn == 'tanh':
        J_matrix = -(np.multiply((y_nn+1),np.log((yhat_nn + 1)/2)/2) + np.multiply((1 - (y_nn+1)/2),np.log(1 - (yhat_nn+1)/2)))
    else: # SIGMOID (same as logistic regression loss function)
        J_matrix = -(np.multiply(y_nn, np.log(yhat_nn)) + np.multiply((1 - y_nn), np.log(1 - yhat_nn)))
        
    J_matrix[np.isnan(J_matrix)] = 0
    J = np.sum(J_matrix)

    # normalizing
    J = J/m

    # adding the regularization term. The bias column is avoided
    # λ*(sum(theta^2))/2m
    theta_square_sum = 0
    for i in np.arange(len(Theta)):
        theta_square_sum += np.sum(np.square(Theta[i]))
    regularization_term = Lambda * theta_square_sum / (2*m)
    J = J + regularization_term 
    
    # Backpropogation---------------------
    # preparing variables
    delta = [[] for i in np.arange(nl+1)] # list to hold error terms of each node on each layer (error terms: derivative on the cost function over the nontransformed input of that node: d(E)/d(z)
    Delta = [[] for i in np.arange(nl)] # list to hold the gradient of each node on each layer (Delta = delta(k)*a(k-1) = error term*transformed output from the previous layer)
    ThetaGrad  =  [[] for i in np.arange(nl)] # normalized gradient plus the regularization term gradient
    grad = np.zeros(snel[-1]+1) # normalized gradient in flatten array
    # computing error terms for all training examples
    # the last layer is special, so separate calc
    delta[nl] = yhat_nn - y_nn ## error term of the output layer; can be derived from the partial derivative function of the J-matrix; exactly equal to the residual in return
    Delta[nl-1] = delta[nl].T * a[nl-1] ## gradient (error term*output from the previous layer)
    ThetaGrad[nl-1] = Delta[nl-1]/m + np.hstack((np.zeros((layers[nl],1)), Theta[nl-1][:, 1:]))*Lambda/m # normalized gradient plus the regularization gradient(bias derivative all zero)
    grad[np.arange((snel[nl-1])+1, snel[nl]+1)] = ThetaGrad[nl-1].flatten()
    # the rest of the layers down to the input layer
    for ii in np.arange(1,nl): # starting from 2nd to last
        bi = nl - ii # backprop index
        delta[bi]= np.multiply((delta[bi + 1] * Theta[bi][:, 1:]), np.vectorize(gF)(z[bi]))        
        Delta[bi-1] = delta[bi].T * a[bi-1] 
        ThetaGrad[bi-1] = Delta[bi-1]/m + np.hstack((np.zeros((layers[bi],1)), Theta[bi-1][:, 1:]))*Lambda/m
        grad[np.arange((snel[bi-1]), snel[bi]+1)] = ThetaGrad[bi-1].flatten()
            
    return J, grad

# gradient descent 
# regarding different types of learning rate: https://www.mygreatlearning.com/blog/understanding-learning-rate-in-machine-learning/

# to do: add gtol stopping criteria
# to do: investigate how to implement stochastic gradient descent instead of full batch gradient on all samples
# opt_options = {"l_rate": 0.1, 'l_type': 'constant','ftol': 10e-6, "maxIter": None}

def fmincg(fun, p0, opt_options = {"l_rate": 0.1, 'l_type': 'constant','ftol': 10e-6, "maxIter": None}, **kwargs):
    """[summary]

    Args:
        fun ([type]): objective function
        p0 (array): initial parameter to optimize on
        options(dictionary):options to configure an optimization algorithm
    """
        
    l_rate0 = opt_options.get('l_rate')
    l_type = opt_options.get('l_type')
    maxIter = opt_options.get('maxIter')
    ftol = opt_options.get('ftol')
    
    if (l_type in ['adaptive-RMSProp']) & (maxIter==None）:
        print("ftol won't be used, please provide max iterations for adaptive learning rate method")
        break
    
    # initial evaluation
    J, grad = fun(p0, **kwargs)
    l_rate = l_rate0
    p = p0 - grad*l_rate
    s = random.uniform(0,1) # momentum initial value for adaptive learning rate method
    
    diag = pd.DataFrame(data = {'iteration':[0], 'objective': [J], 'learning_rate': [l_rate]})    
    
    print("Interation 1: Objective Value = {}".format(J))
    # updating weight in iterations
    i = 1
    while i < int(maxIter):
        J, grad = fun(p, **kwargs)
        if l_type == 'constant':  # approach 1: constant learning rate
            print('apply constant learning rate')
            p = p - grad*l_rate
        elif l_type == 'decayed':
            print('apply decayed learning rate')
            l_rate = l_rate/(1+0.1*i) #  (decay rate: 0.1)
            p = p - grad*l_rate 
        elif l_type == 'scheduled':
            print('apply scheduled learning rate')
            l_rate = l_rate*(0.3**(i//50)) # (how much to reduce learning rate: 0.3, frequency parameter: 50)
            p = p - grad*l_rate 
        elif l_type == 'cycling':
            print('apply scheduled learning rate')
            l_rate_min, l_rate_max = 0.01, l_rate
            S = 10 # step size
            if i < 10:
                l_rate = l_rate_max - (l_rate_max - l_rate_min)/S*(S-i+1)
            else:
                l_rate = l_rate + (l_rate_max - l_rate_min)*((-1)**(i//S+1))
            p = p - grad*l_rate 
        elif l_type == 'adaptive-RMSProp': # widely used in training deep neural networks with stochastic gradient descent 
            print('apply adaptive learning rate')
            s = s*0.8 + (1-0.8)*(grad**2) # (forgetting factor 0.8 - normally between 0.7-0.9)
            l_rate = l_rate0/np.sqrt(s)
            p = p - grad*l_rate
        else:
            print('no specified method available, default to constant')
            p = p - grad*l_rate
        diag = diag.append({'iteration':i,'objective':J, 'learning_rate':l_rate}, ignore_index=True)
        print("Interation {}: Objective Value = {}".format(i+1, J))
        i+=1
        
        if l_type not in ['adaptive']：
            J_current = list(diag['objective'])[-1] 
            J_previous = list(diag['objective'])[-2] 
            if abs(J_current - J_previous) < ftol*(1+abs(J_previous)):
                break

    return J, p, diag
    
    