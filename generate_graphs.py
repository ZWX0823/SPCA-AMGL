import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity

def generate_graphs(X, n_neighbors):
   
    K = 5    #相似度矩阵的个数
    (mn, b) = X.shape
    Si = np.zeros([K,mn,mn])

    C = kneighbors_graph(X, n_neighbors, mode='connectivity', include_self=True).toarray()
    C = (C + C.T) / 2
    C[C < 1] = 0

    for i in range(mn):
        for j in range(i,mn):
            if C[i,j] == 1:
                #这里的相似度矩阵要与K一致
                Si[0,i,j] = Si[0,j,i] = rbf_kernel(np.array([X[i]]), np.array([X[j]]))    #gamma:默认值（1 / X.shape[1]）
                Si[1,i,j] = Si[1,j,i] = cosine_similarity(np.array([X[i]]), np.array([X[j]]))
                Si[2,i,j] = Si[2,j,i] = rbf_kernel(np.array([X[i]]), np.array([X[j]]), gamma=0.5)
                Si[3,i,j] = Si[3,j,i] = rbf_kernel(np.array([X[i]]), np.array([X[j]]), gamma=0.05)
                Si[4,i,j] = Si[4,j,i] = rbf_kernel(np.array([X[i]]), np.array([X[j]]), gamma=0.1)
    
    for i in range(K):
        #行归一化,D是一个对角矩阵,其对角元素为Si[i]行之和
        D = np.diag(np.array([np.sum(j) for j in Si[i]]))
        Si[i] = np.linalg.inv(D) @ Si[i]

    return Si