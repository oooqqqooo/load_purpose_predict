import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import cross_validation

from sklearn.model_selection import GridSearchCV
import numpy as np
#result = pd.DataFrame()
#print(result)
from sklearn import svm
list_para = []
list_score = []
list_num = [10,12,14,16,18,20,22,24,26,28,30]
for i in list_num:
    wo = r'C:\Users\admin\Desktop\file\num_topics= %d.csv' % i
    data = pd.read_csv(wo)
    # print(data.head())
    #print(type(data.iloc[0, 0]))
    #print(type(data.iloc[0, 10]))
    print(data.isnull().any())

    #predict_null = pd.isnull(data['dimension #1'])
    #data_null = data[predict_null == True]
    #print(data_null)
    label_si = data['label']

    print(data.head())
    features_si = data.drop('label', 1)

    #print(features_si.head())
    #print(label_si.head())
    # print(len(data),len(label))
    print(features_si.shape, label_si.shape)

    # X_train, X_test, Y_train, Y_test = train_test_split(features_si, label_si,test_size=0.5)
    # print(type(X_test))

    
    parameters = {
        'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
              18, 19, 20], \
        'kernel': ('rbf', 'linear')}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(features_si, label_si)
    #print(clf.best_params_)
    #print(clf.best_score_)

    list_para.append(clf.best_params_)
    list_score.append(clf.best_score_)
print(list_para)
print(list_score)