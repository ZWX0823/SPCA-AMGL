# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, cosine_similarity

def generate_graphs(X, n_neighbors):
   
    K = 3    #相似度矩阵的个数
    (n, b) = X.shape

    Si = np.zeros([K,n,n])
    Si[0] = rbf_kernel(X)
    Si[1] = laplacian_kernel(X)
    Si[2] = cosine_similarity(X)
    Si[2][Si[2] < 0] = 0
  
    # 构建KNN图
    for k in range(K):
        index_sorted = np.argsort(Si[k])
        for i in range(n):
            for j in index_sorted[i][:-n_neighbors]:
                Si[k,i,j] = 0
    
    # 行归一化,D是一个对角矩阵,其对角元素为Si[i]行之和
    # 经行归一化后矩阵,Si不再是对角矩阵
    for k in range(K):
        D = np.diag(np.sum(Si[k], axis=1))
        Si[k] = np.linalg.inv(D) @ Si[k]
 
    return Si
