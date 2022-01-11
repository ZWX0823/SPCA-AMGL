import numpy as np
import random
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import minmax_scale, normalize

def data_preprocess(data_path, labels_path, W, H):
    '''
    加载数据并对数据进行处理
    参数：
        data_path:数据路径
        labels_path:数据对应的标签路径
        H:截取图像的高
        W:截取图像的宽
    '''
    #注意:所有数据集,数据必须在字典最后一项
    data_mat = loadmat(data_path)
    labels_mat = loadmat(labels_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    labels_index = list(labels_mat)[-1]
    labels_mat = labels_mat[labels_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    # 下面这种方法不行,是有监督的方式了
    
    num_nozero = 0
    while num_nozero / (W * H) < 0.9:
        m1 = random.randint(0, m - H)
        m2 = m1 + H
        n1 = random.randint(0, n - W)
        n2 = n1 + W
        X = data_mat[m1:m2, n1:n2, :]
        y = labels_mat[m1:m2, n1:n2]
        X = X.reshape(-1,X.shape[2])
        y = y.reshape(H * W)
        num_nozero = np.where(y != 0)[0].shape[0]
    '''
    
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
    y = labels_mat[m1:m2, n1:n2]
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(H * W)
   
    index_nozero = np.where(y != 0)[0]
    X = X[index_nozero]
    y = y[index_nozero]
    '''
    
    #数据进行标准化（使用标准化比正则化效果好,且仅在截取的区域上使用标准化比先标准化后截取效果好）
    transformer1 = Normalizer().fit(X)
    X = transformer1.transform(X)
    transformer2 = StandardScaler(with_std=False).fit(X)
    X = transformer2.transform(X)

    if X.shape[0] < 200:
        print('请增大尺寸!')
    elif X.shape[0] > 1200:
        print('请减小尺寸!')

    return X, y

def preprocess_nonzero(data_path, labels_path):
    '''
    最基础的预处理:去除背景元素
    '''
    data_mat = loadmat(data_path)
    labels_mat = loadmat(labels_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    labels_index = list(labels_mat)[-1]
    labels_mat = labels_mat[labels_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    labels_mat = labels_mat.reshape(m * n)
    index_nozero = np.where(labels_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    labels_mat = labels_mat[index_nozero]

    return data_mat, labels_mat

def preprocess_minmaxscale(data_path, labels_path):
    '''
    在去除背景元素的基础上,使用minmax正则化
    '''
    data_mat = loadmat(data_path)
    labels_mat = loadmat(labels_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    labels_index = list(labels_mat)[-1]
    labels_mat = labels_mat[labels_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    labels_mat = labels_mat.reshape(m * n)
    index_nozero = np.where(labels_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    labels_mat = labels_mat[index_nozero]

    data_mat = minmax_scale(data_mat)

    return data_mat, labels_mat

def preprocess_normalize(data_path, labels_path):
    '''
    在去除背景元素的基础上,正则化到[0,1]
    '''
    data_mat = loadmat(data_path)
    labels_mat = loadmat(labels_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    labels_index = list(labels_mat)[-1]
    labels_mat = labels_mat[labels_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    labels_mat = labels_mat.reshape(m * n)
    index_nozero = np.where(labels_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    labels_mat = labels_mat[index_nozero]

    data_mat = normalize(data_mat)

    return data_mat, labels_mat

def preprocess_normal_std(data_path, labels_path):
    data_mat = loadmat(data_path)
    labels_mat = loadmat(labels_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    labels_index = list(labels_mat)[-1]
    labels_mat = labels_mat[labels_index]

    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    labels_mat = labels_mat.reshape(m * n)
    index_nozero = np.where(labels_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    labels_mat = labels_mat[index_nozero]
    
    #数据进行标准化（使用标准化比正则化效果好,且仅在截取的区域上使用标准化比先标准化后截取效果好）
    transformer1 = Normalizer().fit(data_mat)
    data_mat = transformer1.transform(data_mat)
    transformer2 = StandardScaler(with_std=False).fit(data_mat)
    data_mat = transformer2.transform(data_mat)

    return data_mat, labels_mat




