import numpy as np
def featureNormalize(X):
    """returns a normalized version of X where the mean value of each feature is 0 and 
    the standard deviation is 1.

    Args:
        X: raw feature matrix

    Returns:
        X_norm: normalized feature matrix with mean value 0 and standard 
        deviation of 1
    """
    
    # column-wise mean
    mu = np.apply_along_axis(np.mean,0,X)
    
    # column-wise standard deviation
    sigma = np.apply_along_axis(np.std,0,X)

    # normalized matrix
    X_norm = (X-mu)/sigma
    X_norm[np.isnan(X_norm)] = 0
    X_norm[np.isinf(X_norm)] = 0
    
    return X_norm, mu, sigma