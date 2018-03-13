import pandas as pd
import csv
import jieba
from sklearn.svm import SVC

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# C:\Users\admin\Desktop\train_set_CSV.csv
jieba.load_userdict(r'C:\Users\admin\Desktop\dict.txt')

data = pd.read_csv(r'C:\Users\admin\Desktop\samples_set.csv')
# print(data.head())
# print(data)
store_str = []
store_label = []
for i in range(len(data)):
    store_str.append(data.iloc[i, 0])
    if data.iloc[i, 1] == 0:
        store_label.append(-1)
    else:
        store_label.append(1)

print(store_label[5095: 5111])

d = {'label': store_label}
store_label = pd.DataFrame(d)

word_cut = []
for i in store_str:
    seg_list = jieba.cut(i)
    word_cut.append(' '.join(seg_list))
#print(word_cut[:3])
# 分词完应该不用再把词输出一遍了吧？

stop_word_dic = open(r'C:\Users\admin\Desktop\stop_words.txt', 'rb')
print(type(stop_word_dic))
stop_word_content = stop_word_dic.read()
stop_word_list = stop_word_content.splitlines()
stop_word_dic.close()
#从文件中导入停用词表
#将停用词表转换为list


cntVect = CountVectorizer()#stop_words=stop_word_list)
cntTf = cntVect.fit_transform(word_cut)
#print(cntVect.transform())
print(type(cntTf)) #[3293, 8298, 8087, 4068, 10834, 10474, 2088]
t = cntVect.get_feature_names()
for i in [3293, 8298, 8087, 4068, 10834, 10474, 2088]:
    print(t[i])
print(cntVect)
print(type(cntVect))