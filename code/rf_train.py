import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import cross_validation

from sklearn.model_selection import GridSearchCV
import numpy as np
#result = pd.DataFrame()
#print(result)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
list_para = []
list_score = []
list_num = [10,12,14,16,18,20,22,24,26,28,30]
for i in list_num:
    wo = r'C:\Users\admin\Desktop\file\n_component=%d.csv' % i
    data = pd.read_csv(wo)
    # print(data.head())
    #print(type(data.iloc[0, 0]))
    #print(type(data.iloc[0, 10]))

    label_si = data['label']

    print(data.head())
    features_si = data.drop('label', 1)

    print(features_si.head())
    print(label_si.head())
    # print(len(data),len(label))
    print(features_si.shape, label_si.shape)

    # X_train, X_test, Y_train, Y_test = train_test_split(features_si, label_si,test_size=0.5)
    # print(type(X_test))
    '''
    parameters = {
        'n_estimators':[200,300,400,500,600]}
    rf = RandomForestClassifier() 
    #max_depth=10, n_estimators=100)
    clf = GridSearchCV(rf, parameters, cv=5)
    clf.fit(features_si,   label_si)
    #print(clf.best_params_)
    #print(clf.best_score_)
    '''
    parameters = {
        'n_estimators': [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],'max_depth':[3,4,5,6,7,8]}
    gbdt = GradientBoostingClassifier()
    clf = GridSearchCV(gbdt, parameters,cv = 5)
    clf.fit(features_si, label_si)
    list_para.append(clf.best_params_)
    list_score.append(clf.best_score_)
print(list_para)
print(list_score)
