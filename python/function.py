# -*- coding: utf-8 -*-

import numpy as np

def fun(x, B, C):
    '''
    solve the solution of function f(lambda) = sum(lambda3*Cij/((lambda2 / 2) * Bij + lambda ))=1
    '''
    return (np.sum(C[i] / (B[i] + x) for i in range(B.shape[0])) - 1)
