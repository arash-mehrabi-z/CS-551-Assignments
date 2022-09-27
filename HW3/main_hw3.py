#Import 

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from DataLoader import get_data

#Get data
X_train, X_test, y_train, y_test = get_data(True)

all_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Helper functions
def report_metrics(y_true , y_pred, title):
    conf_mat = confusion_matrix(y_true , y_pred)
    precision, recall, fscore, support = score(y_true, y_pred, labels=all_labels)
    accuracy = conf_mat.diagonal()/conf_mat.sum(axis=1)
    accuracy = accuracy.round(5) * 100
    precision = precision.round(2)
    recall = recall.round(2)
    fscore = fscore.round(2)
    
    print("Total Accuracy:", accuracy_score(y_true, y_pred))
    print('accuracy: {}'.format(accuracy))
    print('avg accuracy: {}'.format(np.sum(accuracy)/accuracy.size))
    print('precision: {}'.format(precision))
    print('avg precision: {}'.format(np.sum(precision) / precision.size))
    print('recall: {}'.format(recall))
    print('avg recall: {}'.format(np.sum(recall) / recall.size))
    print('fscore: {}'.format(fscore))
    print('avg fscore: {}'.format(np.sum(fscore) / fscore.size))
    print('support: {}'.format(support))
    
    precision, recall, fscore, support = score(y_true, y_pred, labels=all_labels, average='weighted')
    
    print("weighted Total Accuracy:", accuracy_score(y_true, y_pred))
    print('weighted avg accuracy: {}'.format(np.sum(accuracy)/accuracy.size))
    print('weighted avg precision: {}'.format(np.sum(precision) / precision.size))
    print('weighted avg recall: {}'.format(np.sum(recall) / recall.size))
    print('weighted avg fscore: {}'.format(np.sum(fscore) / fscore.size))
    
    plt.figure(figsize=(16,9))
    sn.heatmap(conf_mat, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(f"results/{title}.png")
    
# Pre-Processing
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)

# KNN Model
n_neighbors = [1, 3, 7]

for n in n_neighbors:
    knn_model = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    knn_model.fit(X_train, y_train)
    
    y_pred_knn = knn_model.predict(X_train)
    title = "KNN with K=" + str(n) + " On the Training Set"
    print('\n', title)
    report_metrics(y_train, y_pred_knn, title)
    
    y_pred_knn = knn_model.predict(X_test)
    title = "KNN with K=" + str(n) + " On the Test Set"
    print('\n', title)
    report_metrics(y_test, y_pred_knn, title)
    
# other hyper-parameter of KNN
knn_model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, weights='distance')
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_train)
title = "KNN with K=1 and weights=distance" + " On the Training Set"
print('\n', title)
report_metrics(y_train, y_pred_knn, title)

y_pred_knn = knn_model.predict(X_test)
title = "KNN with K=1 and weights=distance" + " On the Test Set"
print('\n', title)
report_metrics(y_test, y_pred_knn, title)

# Decision Tree
min_impurity_decreases = [1e-7, 1e-5, 1e-3]

for m in min_impurity_decreases:
    dtree_model = DecisionTreeClassifier(min_impurity_decrease=m)
    dtree_model.fit(X_train , y_train)
    
    y_pred_dtree = dtree_model.predict(X_train)
    title = "Decision Tree with min_impurity_decrease=" + str(m) + " On the Training Set"
    print('\n', title)
    report_metrics(y_train, y_pred_dtree, title)
    
    y_pred_dtree = dtree_model.predict(X_test)
    title = "Decision Tree with min_impurity_decrease=" + str(m) + " On the Test Set"
    print('\n', title)
    report_metrics(y_test, y_pred_dtree, title)
    
# Other hyper-parameter of the DT
m_depths = [28*28, 28*28//10, 28*28//14]

for m in m_depths:
    dtree_model = DecisionTreeClassifier(min_impurity_decrease=1e-7, max_depth=m)
    dtree_model.fit(X_train , y_train)
    
    y_pred_dtree = dtree_model.predict(X_train)
    title = "Decision Tree with max depth=" + str(m) + " On the Training Set"
    print('\n', title)
    report_metrics(y_train, y_pred_dtree, title)
    
    y_pred_dtree = dtree_model.predict(X_test)
    title = "Decision Tree with max depth=" + str(m) + " On the Test Set"
    print('\n', title)
    report_metrics(y_test, y_pred_dtree, title)
    
# Random Forest 
n_estimators = [5, 10, 40, 100]

for n in n_estimators:
    rforest_model = RandomForestClassifier(n_estimators=n)
    rforest_model.fit(X_train , y_train)
    
    y_pred_rforest = rforest_model.predict(X_train)
    title = "Random Forrest with number of trees=" + str(n) + " On the Training Set"
    print('\n', title)
    report_metrics(y_train, y_pred_rforest, title)
    
    y_pred_rforest = rforest_model.predict(X_test)
    title = "Random Forrest with number of trees=" + str(n) + " On the Test Set"
    print('\n', title)
    report_metrics(y_test, y_pred_rforest, title)


