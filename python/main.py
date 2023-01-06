import numpy as np
import math
from numpy.random import rand
from work.optimize_W import optimize_W
from work.optimize_S import optimize_S
from work.optimize_alpha_k import optimize_alpha_k
from work.generate_graphs import generate_graphs
from work.preprocessing import *
from work.evaluate import *
from scipy.linalg import orth

'''
数据集url
'''
data_path = 'work/data/Indian_pines_corrected.mat'
label_path = 'work/data/Indian_pines_gt.mat'
data_mat, label_mat = preprocess_normal_std(data_path, label_path)

def SPCA_AMGL(X, lamda1, lamda2, lamda3, d):
    '''
    X: 数据矩阵,论文中shape为[b,n],代码中实际为[n,b]
    d: 选择的波段子集中包含的波段个数
    '''
    n_neighbors = 6
    
    #正则化子图Si
    Si = generate_graphs(X, n_neighbors)
    K = Si.shape[0]

    #初始化W
    b = X.shape[1]
    W = orth(np.random.randn(b,d))

    #初始化S
    S = np.sum(Si, axis=0) / K

    #初始化多图系数   
    alpha = np.ones(K)
    alpha = alpha / K
    
    iteration = 0
    iter_max = 6
    obj = np.zeros(iter_max)
    while iteration < iter_max:
        
        # update W
        W = optimize_W(X, W, S, lamda1, lamda2, 1)
        
        # update S
        S = optimize_S(X, W, Si, lamda2, lamda3, alpha)
    
        # update alpha
        alpha = optimize_alpha(S, Si)
        
        # object function value
        SS = np.zeros(K)
        for k in range(K):
            index = np.where((Si[k] > 0) & (S > 0))
            SS[k] = (alpha[k] ** 2) * (np.sum(Si[k][index] * np.log(Si[k][index])) - np.sum(Si[k][index] * np.log(S[index])))
            
        D = 0.5 * np.diag(np.sum(S, axis=1) + np.sum(S, axis=0))
        L = D - 0.5 * (S + S.T)
        obj[iteration] = - np.trace(W.T @ X.T @ X @ W) + lamda1 * np.sum(np.linalg.norm(w) for w in W) + lamda2 * np.trace(W.T @ X.T @ L @ X @ W) + lamda3 * np.sum(SS)
        print(obj[iteration])
        iteration = iteration + 1

    score = np.sum(W * W, axis=1)
    index_sorted = np.argsort(score)        #升序排列,需要逆序处理
    index_sorted = index_sorted[-d:][::-1]  #选择后d个

    return index_sorted
  
# 稀疏PCA及自适应多图学习的评估
def fun(d, pattern):
    acc_s = 0
    acc_k = 0
    acc_l = 0
    iteration = 3
    for i in range(iteration):
        w = 20
        h = 20
        X, y = data_preprocess(data_path, label_path, w, h)
        # 参数值为0.001可能导致不收敛
        sorted_index = SPCA_AMGL(X, 1, 1000, 1, d)
        data_mat_new = data_mat[:,sorted_index]
        num = 10
        a = np.zeros(num)
        b = np.zeros(num)
        c = np.zeros(num)
        for i in range(num):
            a[i] = eval_band_SVM(data_mat_new, label_mat, pattern)
            b[i] = eval_band_KNN(data_mat_new, label_mat, pattern)
            c[i] = eval_band_LDA(data_mat_new, label_mat, pattern)
        mean_svm = np.mean(a)
        mean_knn = np.mean(b)
        mean_lda = np.mean(c)
        acc_s += mean_svm
        acc_k += mean_knn
        acc_l += mean_lda
    acc_s = round(acc_s/iteration, 2)
    acc_k = round(acc_k/iteration, 2)
    acc_l = round(acc_l/iteration, 2)
    if pattern == 0:
        print("band_num:{}, acc_svm:{}, acc_knn:{}, acc_lda:{}".format(d, acc_s, acc_k, acc_l))
    elif pattern == 1:
        print("band_num:{}, kappa_svm:{}, kappa_knn:{}, kappa_lda:{}".format(d, acc_s, acc_k, acc_l))
    
for  d in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    fun(d, 0)
