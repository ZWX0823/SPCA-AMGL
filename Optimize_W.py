import numpy as np
from numpy.random import rand

def Optimize_W(X, W_init, S, d, alpha, beta, s):

    b = X.shape[1]
    D = np.diag(np.sum(S, axis=1))
    L = D - S
    row, col = np.diag_indices(b)
    Q = np.zeros([b,b])
    Q[row,col] = np.array([1 / np.linalg.norm(w) for w in W_init])
    A = -X.T @ X + alpha * Q + beta * X.T @ L @ X
    if s==0:
        alpha_A = np.max(np.abs(np.linalg.eig(A)[0]))
    elif s==1:
        ww = rand(b,1)
        for i in range(20):
            m1 = A @ ww
            q = m1 / np.linalg.norm(m1)
            ww = q
        alpha_A = np.abs(ww.T @ A @ ww)
    else:
        print('Warning: error input!!!')
        return;
    
    t = 0
    t_max = 1                            
    A_til = alpha_A * np.eye(b) - A
    T = np.zeros(t_max)
    while t < t_max:
        M = A_til @ W_init
        U, Sigma, VT = np.linalg.svd(M, full_matrices=0)
        W = U @ VT
        T[t] = np.trace(W.T @ A @ W)
        print(T[t])
        if t >= 1:
            err1 = T[t-1] - T[t]
        t = t + 1
            
    return W

    