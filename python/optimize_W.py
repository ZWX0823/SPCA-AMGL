# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import rand

def optimize_W(X, W, S, lambda1, lambda2, s):

    b = X.shape[1]
    D = 0.5 * np.diag(np.sum(S, axis=1) + np.sum(S, axis=0))
    L = D - 0.5 * (S + S.T)
    row, col = np.diag_indices(b)
    Q = np.zeros([b,b])
    Q[row,col] = 1/np.linalg.norm(W, axis=1)
    A = -(X.T @ X) + lambda1 * Q + lambda2 * (X.T @ L @ X)
    #A = lambda1 * Q + lambda2 * (X.T @ L @ X)
    if s==0:
        lambda1_A = np.max(np.abs(np.linalg.eig(A)[0]))
    elif s==1:
        ww = rand(b,1)
        for i in range(20):
            m1 = A @ ww
            q = m1 / np.linalg.norm(m1)
            ww = q
        lambda1_A = np.abs(ww.T @ A @ ww)
    else:
        print('Warning: error input!!!')
        return;
    
    t = 0
    t_max = 15            
    A_til = lambda1_A * np.eye(b) - A
    T = np.zeros(t_max)
    while t < t_max:
        M = A_til @ W
        U, Sigma, VT = np.linalg.svd(M, full_matrices=0)
        W = U @ VT
        T[t] = np.trace(W.T @ A @ W)
        #print("T[t]: ", T[t])
        t = t + 1
    
    return W

    
