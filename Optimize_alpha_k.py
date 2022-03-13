import numpy as np
import math

def Optimize_alpha_k(Si, S):

    # Make sure division makes sense
    eps = 0.0001
    mn = S.shape[0]
    K = Si.shape[0]
    C_k = np.zeros(K)
    for k in range(K):
        index1 = np.where(Si[k] > 0)
        tmp1 = np.sum(Si[k][index1] * np.log(Si[k][index1]))
        index2 = np.where(S > 0)
        tmp2 = np.sum(Si[k][index2] * np.log(S[index2]))
        C_k[k] = tmp1 - tmp2
    tmp3 = np.sum(1 / C_k[k] for k in range(K))
    alpha_k = np.array([(1 / C_k[k]) / tmp3 for k in range(K)])

    return alpha_k

