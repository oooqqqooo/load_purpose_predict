
import pandas as pd
import csv
#import jieba
from sklearn.svm import SVC

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# C:\Users\admin\Desktop\train_set_CSV.csv
#jieba.load_userdict(r'C:\Users\admin\Desktop\dict.txt')

data = pd.read_csv(r'C:\Users\admin\Desktop\train_set_CSV.csv')
# print(data[:3])

print(type(data))
# print(data.iloc[0,]) #输出第 i行
# print(type(data.iloc[0, 1]))  #输出第i行第j列 <class 'numpy.int64'>
# print(type(data.iloc[0, 0]))  #<class 'str'>
length = len(data)

pd_label_0 = data[data['贷款意图判断'].isin([0])]
print(len(pd_label_0))
pd_label_1 = data[data['贷款意图判断'].isin([1])]
print(len(pd_label_1))

print(pd_label_0.iloc[0, 0])
print(pd_label_0.iloc[0, 1])

pd_label_1 = pd_label_1.sample(n=5200) #label = 1 samples 5200
print(pd_label_0.head())
print(pd_label_1.head())
print('----------------')
df = pd.concat([pd_label_0,pd_label_1])
print(df.head)

print(len(pd_label_0), len(df))
print(pd_label_0.head())
# with open(r'C:\Users\admin\Desktop\samples_set.txt', 'w') as file:
#    file.write(df)
df.to_csv(r'C:\Users\admin\Desktop\samples_set.csv',encoding='utf-8',index=False)
'''
print(len(pd_label_1))

store_str = []
store_label = []
for i in range(len(pd_lable_0)):
    store_str.append(pd_lable_0.iloc[i, 0])
    store_label.append(-1)

for i in range(len(pd_label_1)):
    store_str.append(pd_label_1.iloc[i, 0])
    store_label.append(pd_label_1.iloc[i, 1])

print(store_str[:3])
print(store_label[:3])
print(len(store_str) == len(store_label))
word_cut = []
for i in store_str:
    seg_list = jieba.cut(i)
    word_cut.append(' '.join(seg_list))

print(word_cut[:3])


cntVect = CountVectorizer()
cntTf = cntVect.fit_transform(word_cut)

lda = LatentDirichletAllocation(n_components=18, max_iter=100, learning_method='batch')
lda.fit(cntTf)

def print_to_words(model, feature_name, n_top_words):
    for topic_index, topic in enumerate(model.components_):
        print(' topic #%d: ' %  topic_index)
        print(' '.join([feature_name[i] \
                        for i in topic.argsort()[:-n_top_words - 1: -1]]))

n_top_word = 20
tf_feature_names =cntVect.get_feature_names()
print_to_words(lda, tf_feature_names, n_top_word)

doc_topic_dist = lda.transform(cntTf)
# print(doc_topic_dist[0])
# print(lda.perplexity(cntTf))

print(type(doc_topic_dist))
print(len(doc_topic_dist))
y = np.array(store_label)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(doc_topic_dist, y, test_size=0.2)
print(len(X_train), len(X_test))
print(len(Y_train), len(Y_test))
# x_train, y_train 组成train set
# x_test , y_test  组成test set



clf = SVC()
clf.fit(X_train, Y_train)
'''
# print(store_label[0:20])

# print()
# print(store_label[5095 : 5111])
'''
print(len(X_train) == len(X_test))
# print(set(list(X_test)))
acc = clf.score(X_test, Y_test)
print(acc)
'''


