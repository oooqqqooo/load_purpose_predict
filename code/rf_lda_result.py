#lda + random forest：
#parameters = {'n_estimators': [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],\
# 'max_depth':[3,4,5,6,7,8]}
#k = [10,12,14,16,18,20,22,24,26,28,30]
#[{'criterion': 'gini', 'n_estimators': 300}, {'criterion': 'entropy', 'n_estimators': 400}, \
# {'criterion': 'entropy', 'n_estimators': 500}, {'criterion': 'gini', 'n_estimators': 500}, \
# {'criterion': 'gini', 'n_estimators': 80}, {'criterion': 'gini', 'n_estimators': 400}, \
# {'criterion': 'entropy', 'n_estimators': 80}, {'criterion': 'gini', 'n_estimators': 500}, \
# {'criterion': 'gini', 'n_estimators': 100}, {'criterion': 'entropy', 'n_estimators': 100},\
#  {'criterion': 'entropy', 'n_estimators': 80}]
#[0.60761091156198432, 0.5962527909911659, 0.59537908940879525, 0.6120764974274342, 0.58809824288903989,\
#  0.59800019415590722, 0.60654305407242015, 0.60460149500048543, 0.5976118823415203, 0.60654305407242015,\
#  0.60916415881953212]
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
list_num = [10,12,14,16,18,20,22,24,26,28,30]
list_para =[{'criterion': 'gini', 'n_estimators': 300}, {'criterion': 'entropy', 'n_estimators': 400}, \
            {'criterion': 'entropy', 'n_estimators': 500}, {'criterion': 'gini', 'n_estimators': 500}, \
            {'criterion': 'gini', 'n_estimators': 80}, {'criterion': 'gini', 'n_estimators': 400}, \
            {'criterion': 'entropy', 'n_estimators': 80}, {'criterion': 'gini', 'n_estimators': 500}, \
            {'criterion': 'gini', 'n_estimators': 100}, {'criterion': 'entropy', 'n_estimators': 100},\
            {'criterion': 'entropy', 'n_estimators': 80}]

list_result = []
for i in range(len(list_num)):
    wo = r'C:\Users\admin\Desktop\file\n_component=%d.csv' % list_num[i]
    data = pd.read_csv(wo)

    label_si = data['label']
    features_si = data.drop('label', 1)

   # X_train, X_test, Y_train, Y_test = train_test_split(features_si, label_si, test_size=0.3)
    #print(X_train.shape,X_test.shape)
    print(features_si)
    print(label_si)
    clf = RandomForestClassifier(criterion=list_para[i]['criterion'], n_estimators=list_para[i]['n_estimators'])
    clf.fit(features_si, label_si)
    result = clf.oob_score_
    print(result)
    list_result.append(result)
    l = [0.1] * 10
    print(clf.predict([l]))
    break
print(list_result)