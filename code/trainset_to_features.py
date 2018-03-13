
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

def foo(model, feature_names, n_top_words, out):
    for topic_idx, topic in enumerate(model.components_):
        x = ('topic #%d' % topic_idx)
        y ='  '.join([feature_names[i] for i in topic.argsort()[: -n_top_words - 1: -1]])
        xy = x + '\n' +y +'\n'
        #print(type(xy))
        out = out + xy
        #print(out)
    return out
cntVect = CountVectorizer(stop_words=stop_word_list)
cntTf = cntVect.fit_transform(word_cut)


list_numb_topics = [10,12,14,16,18,20,22,24,26,28,30]
list_perplexity = []
for topic_numb in list_numb_topics:
    #topic_numb = 18
    lda = LatentDirichletAllocation(n_components=topic_numb, max_iter=1000, learning_method='batch')
    lda.fit(cntTf)

    dic = {}
    out = ""

    n_top_words = 20
    tf_features_names = cntVect.get_feature_names()
    out = foo(lda, tf_features_names, n_top_words, out)
    # print(out)
    wo = r'C:\Users\admin\Desktop\file\n_topic_numb= %d.txt' % topic_numb
    file = open(wo, 'w')
    file.write(out)
    # print(dic)
    doc_topic_dist = lda.transform(cntTf)
    doc_topic_dist = pd.DataFrame(doc_topic_dist, columns=['topic_#%d' % i for i in range(topic_numb)])
    #doc_topic_dist = pd.join([doc_topic_dist,store_label])
    print(doc_topic_dist.head())
    doc_topic_dist = doc_topic_dist.join(store_label,how='right')
    print(doc_topic_dist.head())
    print(store_label.head())
    wo = r'C:\Users\admin\Desktop\file\n_component=%d.csv' % topic_numb
    doc_topic_dist.to_csv(wo, index=False,encoding='utf-8')
    #print(lda.perplexity(cntTf))
    list_perplexity.append(lda.perplexity(cntTf))

file = open(r'C:\Users\admin\Desktop\test.txt','w')
file.write(str(list_perplexity))
file.close()
