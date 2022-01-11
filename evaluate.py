'''
验证:利用SVM和KNN
'''
import numpy as np
from work.estimate import eval_band_SVM
from work.estimate import eval_band_KNN

def evaluate_python (a, data_mat, labels_mat):
   # 适用于python产生的index
   data_mat_new = data_mat[:,a]
   num = 20
   acc_svm = 0
   acc_knn = 0
   kappa_svm = 0
   kappa_knn = 0
   for i in range(num):
      a, b = eval_band_SVM(data_mat_new, labels_mat)
      c, d = eval_band_KNN(data_mat_new, labels_mat)
      acc_svm += a
      kappa_svm += b
      acc_knn += c
      kappa_knn += d
   
   acc_svm /= num
   acc_knn /= num
   kappa_svm /= num
   kappa_knn /= num
   print('acc by svm:', acc_svm)
   print('acc by knn:', acc_knn)
   print('kappa by svm:', kappa_svm)
   print('kappa by knn:', kappa_knn)

def evaluate_matlab (a, data_mat, labels_mat):
   # 适用于matlab产生的index
   a = a - np.ones_like(a)
   data_mat_new = data_mat[:,a]
   num = 20
   acc_svm = 0
   acc_knn = 0
   kappa_svm = 0
   kappa_knn = 0
   for i in range(num):
      a, b = eval_band_SVM(data_mat_new, labels_mat)
      c, d = eval_band_KNN(data_mat_new, labels_mat)
      acc_svm += a
      kappa_svm += b
      acc_knn += c
      kappa_knn += d

   acc_svm /= num
   acc_knn /= num
   kappa_svm /= num
   kappa_knn /= num
   print('acc by svm:', acc_svm)
   print('acc by knn:', acc_knn)
   print('kappa by svm:', kappa_svm)
   print('kappa by knn:', kappa_knn)
