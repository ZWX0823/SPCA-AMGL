import numpy as np
from numpy.random import rand

def Optimize_W(X, W_init, S, d, lamda1, lamda2, s):

    # The bands nums
    b = X.shape[1]
    # Laplace
    D = np.diag(np.sum(S, axis=1))
    L = D - S
    # Transform 2,1-norm into trace operation
    row, col = np.diag_indices(b)
    Q = np.zeros([b,b])
    Q[row,col] = np.array([1 / np.linalg.norm(w) for w in W_init])

    A = -X.T @ X + lamda1 * Q + lamda2 * X.T @ L @ X
    if s==0:
        lamda1_A = np.max(np.abs(np.linalg.eig(A)[0]))
    elif s==1:
        ww = rand(b,1)
        for i in range(20):
            m1 = A @ ww
            q = m1 / np.linalg.norm(m1)
            ww = q
        lamda1_A = np.abs(ww.T @ A @ ww)
    else:
        print('Warning: error input!!!')
        return
        
    # Our goal is to maximize the value of trace(W.T @ A @ W) and we find that its value reaches the maximum at the first cycle
    t = 0
    t_max = 1                            
    A_til = lamda1_A * np.eye(b) - A
    T = np.zeros(t_max)
    W = W_init
    while t < t_max:
        M = A_til @ W
        U, Sigma, VT = np.linalg.svd(M, full_matrices=0)
        W = U @ VT
        T[t] = np.trace(W.T @ A @ W)
        print(T[t])
        if t >= 1:
            err1 = T[t-1] - T[t]
        t = t + 1
            
    return W

    
