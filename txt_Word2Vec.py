from jpype import *
from pyhanlp import *
import sys
from txt_preprocess import  *
import re
import logging
import os.path
from gensim.models import word2vec
import time

# 将json文件转化成列表  列表中的一个表项代表一条新闻的内容
def Conjsontolist(jsonfile):
    ContList=[]
    with open(jsonfile,"r",encoding='utf-8') as f:
        dicts=json.load(f)
        for item in dicts:
            ContList.append(item['content'])
    return ContList

# 对列表的内容进行分词处理  ContList的每个item相当于一个新闻
def CreatWordList(ContList,CutWordtxt):
    for newtxt in ContList:
        # 调用HanLp_Segment对新闻内容进行分词
        Line_seg=HanLp_Segment(newtxt)
        # 将分词后的结果写入文件
        with open(CutWordtxt,'a',encoding='utf-8') as fw:
            fw.write(Line_seg)

# 使用Word2vec得到每个单词的词向量
def Tibet_Word2Vec(CutWordtxt,Binarypath):
    # 导入训练集
    sentences=word2vec.Text8Corpus(CutWordtxt)
    # 构建词向量  size是特征向量的维度  workers是cup数量  min_count是词频少于min_count次数的单词会被丢弃掉  window：当前词与预测词在一个句子中的最大距离是多少
    model=word2vec.Word2Vec(sentences,size=300,workers=2,min_count=5,window=5)
    # 保存至二进制文件
    model.save(Binarypath)
    



if __name__ == '__main__':
    jsonfile='tibet.json'
    CutWordtxt='tibet.txt'
    Binarypath='TibetWord2vec'
    t1=time.time()
    if not os.path.exists(Binarypath):
        if not os.path.exists(CutWordtxt):
            # 读取json文件
            ContList=Conjsontolist(jsonfile)
            # print(ContList[1])
            # 分词，并将结果写入文件
            CreatWordList(ContList,CutWordtxt)
            t2=time.time()
            print("\n分词花费时间为:%s s" % (t2-t1))
        else:
            t2=t1
        # 训练词向量
        Tibet_Word2Vec(CutWordtxt,Binarypath)
        t3=time.time()
        print("\n训练词向量花费时间为:%s s" %(t3-t2))

    # 加载词向量的二进制文件
    model=word2vec.Word2Vec.load(Binarypath)

    print("与军训最相近的词语是\n")
    result=model.most_similar('军训')
    for word in result:
        print(word[0],word[1])
    print(model['西藏'])
    
    print("校长和老师的相似度为:\n")
    print(model.similarity(u'校长',u'老师'))



    
    









