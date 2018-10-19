from jpype import *
import pandas as pd
from pyhanlp import *
import sys
from txt_preprocess import  *
import re
import logging
import os.path
from gensim.models import word2vec
import time
import logging
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import re

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号


# 对每条新闻内容进行分词和语料清洗
def Process_News(single_new):
    # 调用封装的函数对每条新闻进行分词
    segment_new=HanLp_Segment(single_new)
    # 对分词后的内容进行清洗
    # clean_new=DataClean(segment_new)
    return segment_new


def Read_file(jsonfile,CutWordtxt):
    contents=[]
    # 读取json文件
    with open(jsonfile,'r',encoding='utf-8') as f:
        dicts=json.load(f)
        print("分词中...")
        for item in dicts:
            # 调用分词函数处理每篇文章
            content = Process_News(item['content'])
            contents.append(content)
    with open(CutWordtxt,'w',encoding='utf-8') as f:
        for line in contents:
            f.write(line+'\n')
    return contents


# 语料清洗
def DataClean(StrWords):
    pass
    

# 使用Word2vec得到每个单词的词向量
def Tibet_Word2Vec(CutWordtxt,Binarypath):
    t1=time.time()
    # 导入训练集
    sentences=word2vec.Text8Corpus(CutWordtxt)
    # 构建词向量  size是特征向量的维度  workers是cup数量  min_count是词频少于min_count次数的单词会被丢弃掉  window：当前词与预测词在一个句子中的最大距离是多少
    # PathLineSentences 读取目录下的所有文件 sample表示更高频率的词被随机下采样到所设置的阈值  hs=1表示softmax会被使用
    # sg为训练算法  默认为0对应CBOW算法   1对于skip-gram算法
    model=word2vec.Word2Vec(sentences,size=200,workers=4,min_count=10,window=5,negative=3,sample=0.0001,hs=1)
    # 保存至二进制文件
    model.save(Binarypath)
    t2=time.time()
    print("\n训练词向量花费时间为:%s s" %(t2-t1))







# 提取前200个关键词
def ExtractKeywords(KeyWordPath,CutWordtxt,wordCount=200):
    t1=time.time()
    # 提取前200个关键词
    document=open(CutWordtxt,'r',encoding='utf-8').read()
    # print(document)
    KeywordList=list(HanLP.extractKeyword(document,wordCount))
    KeywordStr=" ".join(KeywordList)
    print(KeywordList)
    with open(KeyWordPath,'w',encoding='utf-8') as fw:
        fw.write(KeywordStr)
    t2=time.time()
    print("提取关键词所花时间为",(t2-t1))
    return KeywordList

# 词向量的可视化
def Plot_tnse_2D(word_vectors,words_list):
    tsne=TSNE(n_components=2,random_state=0,n_iter=10000,perplexity=3)
    np.set_printoptions(suppress=True)
    T=tsne.fit_transform(word_vectors)
    labels=words_list  # 关键词作为标签数据
    plt.figure(figsize=(10,6))
    plt.scatter(T[:,0],T[:,1],c='steelblue',edgecolors='k')
    for label,x,y in zip(labels,T[:,0],T[:,1]):
        plt.annotate(label,xy=(x+1,y+1),xytext=(0,0),textcoords='offset points')
    plt.show()

# 对关键词进行可视化
def KeyWordView(KeyWordPath,CutWordtxt,model,wordCount):
    print("正在执行word2vec的关键词可视化...")
    # 提取关键词
    KeywordList=ExtractKeywords(KeyWordPath,CutWordtxt,wordCount)
    Keywordstr=open(KeyWordPath,'r',encoding='utf-8').read()
    KeywordList=Keywordstr.split(" ")
    words_list=[]
    word_vectors=[]
    for word in KeywordList:
        try:
            result=model.wv.most_similar(positive=[word],topn=15)
            print("关键词:{}".format(word))
            print(result)
            print("-"*100)
            w2v=model.wv[word]
            words_list.append(word)
            word_vectors.append(w2v)
        except:
            print("No word:{}".format(word))
    word_vectors=np.array(word_vectors)
    print(word_vectors)
    print("Total words:", len(words_list), '\tWord Embedding shapes:', word_vectors.shape)
    Plot_tnse_2D(word_vectors,words_list)











