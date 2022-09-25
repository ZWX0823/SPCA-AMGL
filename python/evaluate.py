'''
验证:利用SVM和KNN
'''
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
   前三个方法分别是利用SVM,KNN,LDA计算单次分类的精确度或Kappa系数
'''
def eval_band_SVM(data_mat, labels_mat, pattern):
    
    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    clf = SVC(kernel='rbf', C=10000)
    clf.fit(data_train, labels_train)
    labels_pred = clf.predict(data_test)
    if pattern == 0:
        accuracy = metrics.accuracy_score(labels_test, labels_pred) * 100
    elif pattern == 1:
        accuracy = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(clf, 'work/model/Indian_pines_svm_model.m')

    return accuracy

def eval_band_KNN(data_mat, labels_mat, pattern):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
    neigh.fit(data_train, labels_train)
    labels_pred = neigh.predict(data_test)
    if pattern == 0:
        accuracy = metrics.accuracy_score(labels_test, labels_pred) * 100
    elif pattern == 1:
        accuracy = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_knn_model.m')

    return accuracy

def eval_band_LDA(data_mat, labels_mat, pattern):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    clf = LinearDiscriminantAnalysis()
    clf.fit(data_train, labels_train)
    labels_pred = clf.predict(data_test)
    if pattern == 0:
        accuracy = metrics.accuracy_score(labels_test, labels_pred) * 100
    elif pattern == 1:
        accuracy = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_lda_model.m')

    return accuracy

'''
   后两个方法分别是统计python程序和matlab程序降维后的数据的分类平均精确度或Kappa系数
'''
def evaluate_python (data_mat, labels_mat, pattern, index=None):
   # 适用于python产生的index
   if index is None:
      data_mat_new = data_mat
   else:
      data_mat_new = data_mat[:,index]
   num = 10
   a = np.zeros(num)
   b = np.zeros(num)
   c = np.zeros(num)
   for i in range(num):
      a[i] = eval_band_SVM(data_mat_new, labels_mat, pattern)
      b[i] = eval_band_KNN(data_mat_new, labels_mat, pattern)
      c[i] = eval_band_LDA(data_mat_new, labels_mat, pattern)
   mean_svm = round(np.mean(a),2)
   mean_knn = round(np.mean(b),2)
   mean_lda = round(np.mean(c),2)
   std_svm = round(np.std(a),2)
   std_knn = round(np.std(b),2)
   std_lda = round(np.std(c),2)
   
   return mean_svm, mean_knn, mean_lda, std_svm, std_knn, std_lda

def evaluate_matlab (data_mat, labels_mat, pattern, index=None):
   # 适用于matlab产生的index
   if index is None:
      data_mat_new = data_mat
   else:
      index = index - np.ones_like(index)
      data_mat_new = data_mat[:,index]
   num = 10
   a = np.zeros(num)
   b = np.zeros(num)
   c = np.zeros(num)
   for i in range(num):
      a[i] = eval_band_SVM(data_mat_new, labels_mat, pattern)
      b[i] = eval_band_KNN(data_mat_new, labels_mat, pattern)
      c[i] = eval_band_LDA(data_mat_new, labels_mat, pattern)
   mean_svm = round(np.mean(a),2)
   mean_knn = round(np.mean(b),2)
   mean_lda = round(np.mean(c),2)
   std_svm = round(np.std(a),2)
   std_knn = round(np.std(b),2)
   std_lda = round(np.std(c),2)
   
   return mean_svm, mean_knn, mean_lda, std_svm, std_knn, std_lda
