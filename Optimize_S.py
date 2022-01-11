import numpy as np
import math
from scipy import optimize
from work.function import fun

def Optimize_S(X, W, Si, beta, gamma, alpha_k):

    '''
    优化S
    '''

    eps = 0.0001
    mn, b = X.shape
    tmp_A = np.zeros([mn,mn,b])
    for i in range(mn):
        for j in range(mn):
            if i>j:
                tmp_A[i,j] = X[i] - X[j]
                tmp_A[j,i] = 0 - tmp_A[i,j]
    A = np.linalg.norm(tmp_A @ W, axis=2) ** 2
    A = (beta / 2) * A                                              #提前乘好系数

    B = np.zeros([mn,mn])
    for i in range(mn):
        for j in range(mn):
            B[i,j] = np.sum((alpha_k[k] ** 2) * Si[k,i,j] for k in range(Si.shape[0]))
    B = gamma * B                                                   #提前乘好系数

    S = np.zeros([mn,mn])
    for i in range(mn):
        nonzero_index = np.nonzero(B[i])[0]                         #B第i行中非零元索引
        min_value, min_index = np.min(A[i]), np.argmin(A[i])        #找到A第i行的最小值,索引
        if min_index in nonzero_index:
            try:
                result = optimize.root_scalar(fun, args=(A[i,nonzero_index],B[i,nonzero_index]), bracket=[-min_value + eps, 10000], method='brentq')           #root_scalar求解标量函数的根
            except:
                result = optimize.root_scalar(fun, args=(A[i,nonzero_index],B[i,nonzero_index]), bracket=[-min_value + 0.01 * eps, 10000], method='brentq') 
            lamda = result.root
            S[i,nonzero_index] = B[i,nonzero_index] / (A[i,nonzero_index] + lamda)
        else:
            lamda = -min_value
            tmp = fun(lamda, A[i,nonzero_index],B[i,nonzero_index])
            if tmp <= 0:
                S[i,nonzero_index] = B[i,nonzero_index] / (A[i,nonzero_index] + lamda)
                S[i,min_index] = 1 - np.sum(S[i,nonzero_index])
            else:
                try:
                    result = optimize.root_scalar(fun, args=(A[i,nonzero_index],B[i,nonzero_index]), bracket=[-min_value + eps, 10000], method='brentq')
                except:
                    result = optimize.root_scalar(fun, args=(A[i,nonzero_index],B[i,nonzero_index]), bracket=[-min_value + 0.01 * eps, 10000], method='brentq')
                lamda = result.root
                S[i,nonzero_index] = B[i,nonzero_index] / (A[i,nonzero_index] + lamda)
    
    return S













