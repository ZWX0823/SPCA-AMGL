# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import optimize
from work.function import fun

def optimize_S(X, W, Si, lambda2, lambda3, alpha):

    '''
    优化S
    '''

    eps = 0.00001
    n, b = X.shape
    K = len(alpha)
    tmp_B = np.zeros([n,n,b])
    for i in range(n):
        for j in range(n):
            if i>j:
                tmp_B[i,j] = X[i] - X[j]
                tmp_B[j,i] = 0 - tmp_B[i,j]
    B = np.linalg.norm(tmp_B @ W, axis=2) ** 2
    B = (lambda2 / 2) * B                                              #提前乘好系数

    C = np.zeros([n,n])
    for k in range(K):
        C += (alpha[k] ** 2) * Si[k]
    #for i in range(n):
    #    for j in range(n):
    #        C[i,j] = np.sum((alpha[k] ** 2) * Si[k,i,j] for k in range(Si.shape[0]))
    C = lambda3 * C                                                   #提前乘好系数

    S = np.zeros([n,n])
    for i in range(n):
        nonzero_index = np.nonzero(C[i])[0]                         #B第i行中非零元索引
        min_value, min_index = np.min(B[i]), np.argmin(B[i])        #找到A第i行的最小值,索引
        if min_index in nonzero_index:
            try:
                result = optimize.root_scalar(fun, args=(B[i,nonzero_index],C[i,nonzero_index]), bracket=[-min_value + eps, 10000], method='brentq')           #root_scalar求解标量函数的根
            except:
                result = optimize.root_scalar(fun, args=(B[i,nonzero_index],C[i,nonzero_index]), bracket=[-min_value + 0.01 * eps, 10000], method='brentq') 
            lamda = result.root
            S[i,nonzero_index] = C[i,nonzero_index] / (B[i,nonzero_index] + lamda)
        else:
            lamda = -min_value
            tmp = fun(lamda, B[i,nonzero_index],C[i,nonzero_index])
            if tmp <= 0:
                S[i,nonzero_index] = C[i,nonzero_index] / (B[i,nonzero_index] + lamda)
                S[i,min_index] = 1 - np.sum(S[i,nonzero_index])
            else:
                try:
                    result = optimize.root_scalar(fun, args=(B[i,nonzero_index],C[i,nonzero_index]), bracket=[-min_value + eps, 10000], method='brentq')
                except:
                    result = optimize.root_scalar(fun, args=(B[i,nonzero_index],C[i,nonzero_index]), bracket=[-min_value + 0.01 * eps, 10000], method='brentq')
                lamda = result.root
                S[i,nonzero_index] = C[i,nonzero_index] / (B[i,nonzero_index] + lamda)

    return S

