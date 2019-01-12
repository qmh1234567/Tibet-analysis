import sys
from pyhanlp import *
import numpy as np
import gensim
from collections import Counter
import pandas as pd #引入它主要是为了更好的显示效果
from gensim.models import word2vec
import jieba
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import re
sys.path.append(r'../common/')
from txt_Word2Vec import Read_file,Tibet_Word2Vec
from txt_preprocess import LoadWordList,TF_IDF,Sentences_list
import shutil
import codecs
Max_keyword_count=30

'''idf提取关键词
keywordCount:关键词数量
输出：每一篇文章的关键词列表（二级列表）
'''
def TF_IDF_keyword(CutWordtxt,keywordCount=5):
    content_List=LoadWordList(CutWordtxt)
    sentencelist=Sentences_list(content_List) # 二级列表
    ## 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    tf_vectorizer=TfidfVectorizer()  
    keywordlists=[]
    for itemlist in sentencelist:
        # 防止列表为空列表，报错
        try:
            # 将文本转化为词频矩阵
            tf_matrix=tf_vectorizer.fit_transform(itemlist)
        except:
            print(itemlist)
        # 获取词袋模型的所有词语
        word_dict=tf_vectorizer.get_feature_names()
        word=np.array(word_dict)
        # 得到tfidf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
        weight=tf_matrix.toarray()
        word_index=np.argsort(-weight)  
        keyword=word[word_index]
        # 关键字最小长度
        min_length=min(len(x) for x in keyword)
        len_new="".join(itemlist)
        tags=[]
        # i 代表关键字长度 keyword是关键词列表，里面有很多关键词
        for i in range(keywordCount if keywordCount<min_length else min_length):
            tags.append(keyword[0][i])
        keywordlists.append(tags)
    return keywordlists


#  hanlp 提取关键词
def Hanlp_keyword(content_List,CutWordtxt,Keywordtxt):
    print("正在提取关键词..")
    # 统计最大长度的新闻
    max_len=max(len(x) for x in content_List)
    # 提取每篇文章的关键字
    Keywordlists=[]
    for single_new in content_List:
        keywordList=HanLP.extractKeyword(single_new,5)
        Keywordlists.append(keywordList)
    with open(Keywordtxt,'w',encoding='utf-8') as f:
          for itemlist in Keywordlists:
                for strword in itemlist:
                    f.write(strword+" ")
                f.write("\n")



'''将二级列表写入文件
keywordlists:二级列表
keywordfile:文件路径
'''
def write_lists_to_file(keywordlists,keywordfile):
    # 设置newline，否则两行之间会空一行
    # with open(keywordfile,'w',newline='') as f:
    #     csv.writer(f).writerows(keywordlists)
    with open(keywordfile,'w',encoding='utf-8') as f:
        for item in keywordlists:
            item_str=" ".join(item)
            f.write(item_str+"\n")
    print("关键词写入文件成功")
    return True


# 将一份txt文件的每一行都写入一个txt文件
def Write_to_many_txt(Cutwordtxt,filepath,titles):
    with open(Cutwordtxt,'r',encoding='utf-8') as f:
        news_list=f.read().splitlines()
    # 遍历所有新闻
    for i in range(len(titles)):
        # 使用正则表达式去除标题的特殊符号和空格
        titles[i]=re.sub(r'[\s | / \d ＂＂ " ? : “ ” \. \- +]','',titles[i])
        filename=titles[i]+'.txt'
        file=os.path.join(filepath,filename)  # 创建路径
        # os.mknod(file)  # 生成文件
        with open(file,'w',encoding='utf-8') as f:
            f.write(news_list[i])
    print("写入多个txt文件成功")
    



# if __name__ == '__main__':
#     #    word2vec_keyword()
#     jsonfile='./../../Resources/jsonfiles/society.json'   
#     CutWordtxt='./../../Resources/CutWordPath/society.txt'
#     keywordfile='./society_keyword1.txt'
#     keyfile='./society_hanlp.txt'
   

#     # 读取json文件，不去停用词，保留。
#     titles,contents=Read_file(jsonfile,CutWordtxt,flag_stop=False)
  
#     # # 使用TF-IDF 提取关键词
#     keywordlists=TF_IDF_keyword(CutWordtxt)
#     write_lists_to_file(keywordlists,keywordfile)
    
    # # hanlp提取关键词
    # Hanlp_keyword(contents,CutWordtxt,keyfile)




# # 已经训练好的词向量模型
# model = gensim.models.word2vec.Word2Vec.load(Binaryfile)
# # 用word2vec 提取的关键词
# def word2vec_keyword():
#     #此函数计算某词对于模型中各个词的转移概率p(wk|wi)
#     def predict_proba(model,oword, iword):
#         # 获得输入词的词向量
#         iword_vec = model[iword]  
#         # 获取保存权重的词的词库
#         oword = model.wv.vocab[oword]
#         oword_l = model.syn1[oword.point].T
#         dot = np.dot(iword_vec, oword_l)
#         lprob = -sum(np.logaddexp(0, -dot) + oword.code*dot) 
#         return lprob
#     # 各个词对某词wi转移概率的乘积即为p(content|wi)
#     # 如果p(content|wi)越大就说明在出现wi这个词的条件下，此内容概率越大，
#     #那么把所有词的p(content|wi)按照大小降序排列，越靠前的词就越重要，越应该看成是本文的关键词
#     def keywords(s,model):
#         s = [w for w in s if w in model]
#         ws = {w:sum([predict_proba(model,u, w) for u in s]) for w in s}
#         return Counter(ws).most_common()
#     # 将分词后的文件读取为列表
#     with open(CutWordtxt,'r',encoding='utf-8') as f:
#         contents=f.read().splitlines()
#     keyword_word2vec=[]
#     for single_new in contents:
#         x = pd.Series(keywords(jieba.cut(single_new),model))
#         #输出最重要的前5个词
#         keyword_word2vec.append(x[0:5])
#         # print (x[0:5])
#     with open("keyword_word2vec1.txt",'w',encoding='utf-8') as f:
#         for item in keyword_word2vec:
#             f.write(str(item)+"\n")