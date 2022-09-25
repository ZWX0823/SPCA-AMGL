# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale, normalize, minmax_scale

def data_preprocess(data_path, label_path, W, H):
    '''
    加载数据并对数据进行处理
    参数：
        data_path:数据路径
        label_path:数据对应的标签路径
        H:截取图像的高
        W:截取图像的宽
    '''
    #注意:所有数据集,数据必须在字典最后一项
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]

    m, n = data_mat.shape[0], data_mat.shape[1]

    if m > H:
        m1 = random.randint(0, m - H)
        m2 = m1 + H
    else:
        m1 = 0
        m2 = m + 1
    if n > W:
        n1 = random.randint(0, n - W)
        n2 = n1 + W
    else:
        n1 = 0
        n2 = n + 1
   
    # 截取连续区域
    X = data_mat[m1:m2, n1:n2, :]
    y = label_mat[m1:m2, n1:n2]
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(H * W)

    X = normalize(X)
    n, b = X.shape
    
    return X, y

def preprocess_normal_std(data_path, label_path):
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]
    
    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    label_mat = label_mat.reshape(m * n)

    index_nozero = np.where(label_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    label_mat = label_mat[index_nozero]
    
    data_mat = normalize(data_mat)

    return data_mat, label_mat   

def preprocess_nonzero(data_path, label_path):
    '''
    最基础的预处理:去除背景元素
    '''
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    label_mat = label_mat.reshape(m * n)
    index_nozero = np.where(label_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    label_mat = label_mat[index_nozero]

    return data_mat, label_mat

def preprocess_minmaxscale(data_path, label_path):
    '''
    使用minmax正则化
    '''
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    X = data_mat.reshape(m * n, b)
    y = label_mat.reshape(m * n)
    
    X = minmax_scale(X)

    index_nozero = np.where(y != 0)[0]
    data_mat = X[index_nozero]
    label_mat = y[index_nozero]

    return X, y, data_mat, label_mat

def preprocess_normalize(data_path, label_path):
    '''
    正则化到[0,1]
    '''
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    X = data_mat.reshape(m * n, b)
    y = label_mat.reshape(m * n)

    X = normalize(X)

    index_nozero = np.where(y != 0)[0]
    data_mat = X[index_nozero]
    label_mat = y[index_nozero]

    return data_mat, y, data_mat, label_mat

# 除以矩阵的最大值缩放
def preprocess_maxscale(data_path, label_path):
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    label_mat = label_mat.reshape(m * n)
    data_mat = data_mat / np.max(data_mat)
    index_nozero = np.where(label_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    label_mat = label_mat[index_nozero]

    return data_mat, label_mat

# 矩阵的每列除以该列的最大值缩放
def preprocess_col_maxscale(data_path, label_path):
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    label_mat = label_mat.reshape(m * n)
    data_mat = data_mat / np.max(data_mat, axis=0)
    index_nozero = np.where(label_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    label_mat = label_mat[index_nozero]

    return data_mat, label_mat
