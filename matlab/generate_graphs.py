import numpy as np
from sklearn.metrics.pairwise import *

def generate_graphs(X, n_neighbors):

    K = 3   #相似度矩阵个数
    (n, b) = X.shape
    Si[:, :, 0] = rbf_kernel(X)
    Si[:, :, 1] = laplacian_kernel(X)
    Si[:, :, 2] = cosine_similarity(X)
    Si[:, :, 2][Si[:, :, 2] < 0] = 0

    #构建KNN图
    for k in range(K):
        index_sorted = np.argsort(Si[:, :, k])
        for i in range(n):
            for j in index_sorted[i][:-n_neighbors]:
                Si[i, j, k] = 0

    #行归一化
    for k in range(K):
        D = np.diag(np.sum(Si[:, :, k], axis=1))
        Si[:, :, k] = np.linalg.inv(D) @ Si[:, :, k]
    
    return Si

