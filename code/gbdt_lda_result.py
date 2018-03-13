#parameters = {
# 'n_estimators': [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],、
# 'max_depth':    [3,4,5,6,7,8]}
#k = [10,12,14,16,18,20,22,24,26,28,30]
#以上为交叉验证的超参数

#最优参数：
# [{'max_depth': 3, 'n_estimators': 20}, {'max_depth': 3, 'n_estimators': 30}, {'max_depth': 3, 'n_estimators': 80},\
#  {'max_depth': 4, 'n_estimators': 20}, {'max_depth': 4, 'n_estimators': 70}, {'max_depth': 3, 'n_estimators': 80},\
#  {'max_depth': 3, 'n_estimators': 80}, {'max_depth': 3, 'n_estimators': 120}, {'max_depth': 3, 'n_estimators': 100}, \
# {'max_depth': 4, 'n_estimators': 50}, {'max_depth': 4, 'n_estimators': 80}]
#[0.64119988350645574, 0.63634598582661872, 0.63459858266187752, 0.64595670323269583,\
#  0.62110474711193087, 0.63547228424424818, 0.63935540238811761, 0.64294728667119694,\
#  0.64100572759926222, 0.64178235122803606, 0.6471216386758567]
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
list_num = [10,12,14,16,18,20,22,24,26,28,30]
list_para = [{'max_depth': 3, 'n_estimators': 20}, {'max_depth': 3, 'n_estimators': 30}, {'max_depth': 3, 'n_estimators': 80},\
             {'max_depth': 4, 'n_estimators': 20}, {'max_depth': 4, 'n_estimators': 70}, {'max_depth': 3, 'n_estimators': 80},\
             {'max_depth': 3, 'n_estimators': 80}, {'max_depth': 3, 'n_estimators': 120}, {'max_depth': 3, 'n_estimators': 100}, \
             {'max_depth': 4, 'n_estimators': 50}, {'max_depth': 4, 'n_estimators': 80}]

list_result = []
for i in range(len(list_num)):
    wo = r'C:\Users\admin\Desktop\file\n_component=%d.csv' % list_num[i]
    data = pd.read_csv(wo)

    label_si = data['label']
    features_si = data.drop('label', 1)

    X_train, X_test, Y_train, Y_test = train_test_split(features_si, label_si, test_size=0.3)
    print(X_train.shape,X_test.shape)
    clf = GradientBoostingClassifier(max_depth=list_para[i]['max_depth'], n_estimators=list_para[i]['n_estimators'])
    clf.fit(X_train, Y_train)
    result = clf.score(X_test,Y_test)
    print(result)
    list_result.append(result)


print(list_result)
