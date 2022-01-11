import joblib
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def eval_band_SVM(data_mat, labels_mat):
    
    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9)

    #模型训练与拟合
    clf = SVC(kernel='rbf', C=10000)
    clf.fit(data_train, labels_train)
    labels_pred = clf.predict(data_test)
    accuracy1 = metrics.accuracy_score(labels_test, labels_pred) * 100
    accuracy2 = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(clf, 'work/model/Indian_pines_svm_model.m')

    return accuracy1, accuracy2

def eval_band_KNN(data_mat, labels_mat):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9)

    #模型训练与拟合
    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(data_train, labels_train)
    labels_pred = neigh.predict(data_test)
    accuracy1 = metrics.accuracy_score(labels_test, labels_pred) * 100
    accuracy2 = metrics.cohen_kappa_score(labels_test, labels_pred) * 100
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_knn_model.m')

    return accuracy1, accuracy2


