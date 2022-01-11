import numpy as np

def MVPCA (X,d):
    '''
    X.shape = [b,n], where b is the number of bands and n is the number of samples
    '''
    b,n = X.shape
    for i in range(b):
        X[i,:] = X[i,:] - np.mean(X[i,:])
    XXT = X @ X.T
    rank_val = np.diag(XXT)
    rank_val = np.argsort(rank_val)[-d:][::-1]
    return rank_val