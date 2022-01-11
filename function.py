import numpy as np

def fun(x, a, b):
    '''
    求解函数f(lamda) = 求和(gamma * Bij / ((beta / 2) * Aij + lamda )) = 1的根
    '''
    return (np.sum(b[i] / (a[i] + x) for i in range(a.shape[0])) - 1)
