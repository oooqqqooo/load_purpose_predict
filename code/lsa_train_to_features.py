import gensim

import pandas as pd
import csv
import jieba
from sklearn.svm import SVC
import copy
from gensim import corpora,models
from collections import defaultdict
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# C:\Users\admin\Desktop\train_set_CSV.csv
jieba.load_userdict(r'C:\Users\admin\Desktop\dict.txt')

data = pd.read_csv(r'C:\Users\admin\Desktop\samples_set.csv',encoding='utf-8')
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
del store_label[2987]
d = {'label': store_label}
store_label = pd.DataFrame(d)

word_cut = []
index = 0
test = []
for i in store_str:
    seg_list = jieba.lcut(i)
    #word_cut.append(' '.join(seg_list))
    word_cut.append(seg_list)


#print(word_cut[:3])
# 分词完应该不用再把词输出一遍了吧？

stop_word_dic = open(r'C:\Users\admin\Desktop\stop_words.txt',encoding='utf-8')
print(type(stop_word_dic))
print(stop_word_dic)

stop_word_content = stop_word_dic.read()
stop_word_list = stop_word_content.splitlines()
stop_word_list = set(stop_word_list)
#print('#' in stop_word_list)

#test
word_cut = [[word for word in t if word not in stop_word_list] for t in word_cut]

print(word_cut[:2])
frequency = defaultdict(int)
for wow in word_cut:
    for w in wow:
        frequency[w] += 1

word = [[w for w in wow if frequency[w] > 1] for wow in word_cut]

del word[2987]
length_0 = []
for i in range(len(word)):
    if len(word[i]) == 0:
        length_0.append(i)


print(length_0)
dictionary = corpora.Dictionary(word)

#将文档存入字典， 字典有很多功能 比如
#diction.token2id 存放的是单词-id key-value 对
#diction.dfs 存放的 是单词出现的频率

# dictionary.save('file path')
corpus = [dictionary.doc2bow(w) for w in word]

tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]

l = [10,12,14,16,18,20,22,24,26,28,30]
for d in l:
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=d)
    corpus_lsi = lsi_model[corpus_tfidf]
    #print('------')
    store = []
    index = 0
    for doc in corpus_lsi:
        l = []
        if index == 2987:
            print(doc)

        for ii in doc:
            l.append(ii[1])
        store.append(l)
        index += 1
    print(store[2985:2990])

    store[2985:2990]
    print(store[0])
    #s = lsi_model.print_topics()
    #print(s)

    df = pd.DataFrame(store, columns=['dimension #%d' % ii for ii in range(d)])

    df = df.join(store_label, how='right')
    print(df.head())
    print(df.shape)
    wo = r'C:\Users\admin\Desktop\file\num_topics= %d.csv' % d
    df.to_csv(wo, index=False, encoding='utf-8')