import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
list_num = [10,12,14,16,18,20,22,24,26,28,30]
list_para = [{'C': 0.1, 'kernel': 'rbf'}, {'C': 20, 'kernel': 'rbf'}, {'C': 20, 'kernel': 'rbf'},\
        {'C': 13, 'kernel': 'linear'}, {'C': 7, 'kernel': 'linear'}, {'C': 17, 'kernel': 'rbf'},\
        {'C': 7, 'kernel': 'linear'}, {'C': 19, 'kernel': 'linear'}, {'C': 5, 'kernel': 'linear'}, \
        {'C': 12, 'kernel': 'linear'}, {'C': 3, 'kernel': 'linear'}]

#list_score = [0.62925929521405688, 0.62440539753421997, 0.62226968255509174, \
# 0.6327541015435395, 0.61498883603533638, 0.61372682263857881, \
# 0.63537520629065136, 0.62275507232307548, 0.62877390544607314,\
#  0.63197747791476555, 0.63673429764100575]
list_result = []
for i in range(len(list_num)):
    wo = r'C:\Users\admin\Desktop\file\num_topics= %d.csv' % list_num[i]
    data = pd.read_csv(wo)

    label_si = data['label']
    features_si = data.drop('label', 1)

    X_train, X_test, Y_train, Y_test = train_test_split(features_si, label_si, test_size=0.3)
    print(X_train.shape,X_test.shape)
    clf = SVC(kernel=list_para[i]['kernel'], C=list_para[i]['C'])
    clf.fit(X_train, Y_train)
    result = clf.score(X_test,Y_test)
    print(result)
    list_result.append(result)


print(list_result)