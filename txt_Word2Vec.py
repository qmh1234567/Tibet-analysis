from jpype import *
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

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

# 将json文件转化成列表  列表中的一个表项代表一条新闻的内容
def ConjsontoList(jsonfile):
    ContentList=[]
    with open(jsonfile,"r",encoding='utf-8') as f:
        dicts=json.load(f)
        for item in dicts:
            ContentList.append(item['content'])
    return ContentList

# 对列表的内容进行分词处理  ContList的每个item相当于一个新闻
def SegmentContList(Contlist,CutWordtxt):
    t1=time.time()
    StrWords=""
    # 遍历每一条新闻
    for item in Contlist:
        # 调用封装的函数
        content=HanLp_Segment(item)
        StrWords+=str(content)
    t2=time.time()
    print("\n分词花费时间为:%s s" % (t2-t1))
    # 将分词后的结果写入文件
    with open(CutWordtxt,'w',encoding='utf-8') as fw:
        fw.write(StrWords)
    
# 使用Word2vec得到每个单词的词向量
def Tibet_Word2Vec(CutWordpath,Binarypath):
    t1=time.time()
    # 构建词向量  size是特征向量的维度  workers是cup数量  min_count是词频少于min_count次数的单词会被丢弃掉  window：当前词与预测词在一个句子中的最大距离是多少
    # PathLineSentences 读取目录下的所有文件 sample表示更高频率的词被随机下采样到所设置的阈值  hs=1表示softmax会被使用
    model=word2vec.Word2Vec(word2vec.PathLineSentences(CutWordpath),size=300,workers=4,min_count=10,window=5,negative=3,sample=0.001,hs=1)
    # 保存至二进制文件
    model.save(Binarypath)
    t2=time.time()
    print("\n训练词向量花费时间为:%s s" %(t2-t1))

# 词向量的测试
def Test_Word2vec(model):
    # 加载词向量的二进制文件
    model=word2vec.Word2Vec.load(Binarypath)
    print("与国歌最相近的词语是\n")
    result=model.wv.most_similar(positive='国歌',topn=15)
    for word in result:
        print(word[0],word[1])
    print("书记和领导的相似度为:\n")
    print(model.similarity(u'书记',u'领导'))


# 提取前200个关键词
def ExtractKeyword(wordCount=200):
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
def KeyWordView(model,wordCount):
    # 提取关键词
    KeywordList=ExtractKeyword(wordCount)
    Keywordstr=open(KeyWordPath,'r',encoding='utf-8').read()
    KeywordList=Keywordstr.split(" ")
    print(KeywordList)
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


if __name__ == '__main__':
    # 打印信息
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    # 文件的路径
    jsonfile='Resources/tibet.json'
    CutWordtxt='Resources/CutWordPath/tibet.txt'
    Binarypath='Resources/TibetWord2vec'
    CutWordPath='Resources/CutWordPath/'
    KeyWordPath='Resources/keyword.txt'
    # 判断文件是否更新有待完成
    if not os.path.exists(Binarypath):
        if not os.path.exists(CutWordtxt):
            # 读取json文件
            Contlist=ConjsontoList(jsonfile)
            print("正在分词中..")
            # 分词，并将结果写入文件
            SegmentContList(Contlist,CutWordtxt)
            # 训练词向量
            Tibet_Word2Vec(CutWordPath,Binarypath)
    # 加载词向量的二进制文件
    model=word2vec.Word2Vec.load(Binarypath)
    # 词向量可视化
    KeyWordView(model,120)









