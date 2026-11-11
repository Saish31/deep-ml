import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    n = X.shape[0] #rows
    idx = np.random.permutation(n) #indexes shuffled
    
    return X[idx], y[idx]