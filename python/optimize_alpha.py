# -*- coding: utf-8 -*-
import numpy as np
import math

def Optimize_alpha(S, Si):

    eps = 0.00001
    mn = S.shape[0]
    K = Si.shape[0]
    C = np.zeros(K)
    for k in range(K):
        index = np.where((Si[k] > 0) & (S > 0))
        C[k] = np.sum(Si[k][index] * np.log(Si[k][index])) - np.sum(Si[k][index] * np.log(S[index])) + eps
    tmp = np.sum(1 / C[k] for k in range(K))
    alpha = np.array([(1 / C[k]) / tmp for k in range(K)])

    return alpha
