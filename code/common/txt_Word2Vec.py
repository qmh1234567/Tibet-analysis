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
from collections import Counter

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号


'''对每条新闻内容进行分词和语料清洗'''
def Process_News(single_new,flag_stop=True):
    # 调用封装的函数对每条新闻进行分词
    segment_new=HanLp_Segment(single_new,flag_stop)
    # 对分词后的内容进行清洗
    clean_new=DataClean(segment_new)
    return segment_new





'''读取文件  
jsonfile：json文件路径 CutWordtxt：分词后的文件路径 
contents:列表 每一项是分好词的新闻
'''
def Read_file(jsonfile,CutWordtxt,flag_stop=True):
    contents=[]
    # 读取json文件
    with open(jsonfile,'r',encoding='utf-8') as f:
        dicts=json.load(f)
        print("分词中...")
        for item in dicts:
            # 调用分词函数处理每篇文章
            content = Process_News(item['content'],flag_stop)
            # 调用语料清洗函数
            content=content[0:599]
            content=DataClean(content)
            contents.append(content)
    with open(CutWordtxt,'w',encoding='utf-8') as f:
        for line in contents:
            f.write(line+'\n')
    return contents




'''数据清洗  返回清理后的内容'''
def DataClean(content):
    # 正则表达式
    RegExp=[
            "新闻( )?(图片 )?来源.+?\n",
            r'\d{4}( )?年[度初底末]',
            r'\d{4}年\d+月\d+ [日晚]',
            r'\d{4}( )?年\d+月(\d+[日号])?(\d+[时点])?(\d+分)?',
            r'\d{4}( )?年(\d+)?',
            r'\d+月\d+-\d+[日号]',
            r'\d+月( )?\d+[日号](\d+[时点])?(\d+分)?',
            r'\d+日',
            r'(今年)?\d+月(\d+)?',
            r'(上午)?\d+[时点](\d+分)?',
            r'([上下]午)?\d+:\d+',
            '[A-Za-z][0-9]+',
            '[A-Za-z_&#*()+=.:：【】"“”]'
           ]

    # 循环匹配去除
    for i in RegExp:
        if i[len(i)-1:] == '\n':
            content = re.sub(i,'\n', content)
        else:
            content = re.sub(i,'', content)
    return content
    

'''使用Word2vec得到每个单词的词向量
CutWordtxt: 分词后的文件路径
Binaryfile：保存word2vec处理后的词向量的二进制文件
'''
def Tibet_Word2Vec(CutWordtxt,Binaryfile):
    t1=time.time()
    # 导入训练集
    sentences=word2vec.Text8Corpus(CutWordtxt)
    # 构建词向量  size是特征向量的维度  workers是cup数量  min_count是词频少于min_count次数的单词会被丢弃掉  window：当前词与预测词在一个句子中的最大距离是多少
    # PathLineSentences 读取目录下的所有文件 sample表示更高频率的词被随机下采样到所设置的阈值  hs=1表示softmax会被使用
    # sg为训练算法  默认为0对应CBOW算法   1对于skip-gram算法
    model=word2vec.Word2Vec(sentences,size=128,workers=4,min_count=10,window=4,negative=3,sample=0.0001,hs=1)
    # 保存至二进制文件
    model.save(Binaryfile)
    t2=time.time()
    print("\n训练词向量花费时间为:%s s" %(t2-t1))
    return model


# def predict_proba(model,oword,iword):
#     iword_vec = model[iword]
#     oword = model.wv.vocab[oword]
#     oword_l = model.syn1[oword.point].T
#     dot = np.dot(iword_vec, oword_l)
#     lprob = -sum(np.logaddexp(0, -dot) + oword.code*dot) 
#     return lprob

# # 提取关键词
# def Extract_keywords(s,model):
#     s = [w for w in s if w in model]
#     ws = {w:sum([predict_proba(model,u, w) for u in s]) for w in s}
#     return Counter(ws).most_common()



'''在二维图中显示词向量
传入参数 词向量和单词列表
'''
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


'''
对一些词进行可视化
KeywordList：关键词列表
model：word2vec加载后的model
'''
def Wordlist_View(wordlist,model,topn):
    print("正在执行word2vec的关键词可视化...")
    # 提取关键词
    words_list=[]
    word_vectors=[]
    for word in wordlist:
        try:
            result=model.wv.most_similar(positive=[word],topn=topn)
            print("关键词:{}".format(word))
            print(result)
            print("-"*100)
            # 现将关键词的词向量添加进去
            w2v=model.wv[word]
            word_vectors.append(w2v)
            words_list.append(word)
            # 得到意思相近的词向量
            for item in result:
                word_vectors.append(model.wv[item[0]]) # 再添加相关词的词向量
                words_list.append(item[0])
        except:
            print("No word:{}".format(word))
    word_vectors=np.array(word_vectors)
    print("Total words:", len(words_list), '\tWord Embedding shapes:', word_vectors.shape)
    Plot_tnse_2D(word_vectors,words_list)





    





