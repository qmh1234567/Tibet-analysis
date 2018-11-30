from jpype import *
import os
from pyhanlp import *
import time
import json
# import matplotlib.pyplot as plt
import pylab
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
import jieba.posseg as psg
'''创建停用词表'''
def stopwordslist(path):
    stpwds=open(path,"r",encoding='utf-8').read()
    # 将文件内容转化为列表
    stplst=stpwds.splitlines()
    return stplst




'''
HanLp 分词函数
raw代表一条新闻内容 flag_stop=True表示去停用词
str_words：分词后的字符串
'''
def HanLp_Segment(raw,flag_stop=True):
    # 停用词列表
    stop_word_path='../../Resources/stopwords.txt'
    stopwordlist=stopwordslist(stop_word_path)
    #Hanlp分词
    # NLPTokenizer=JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
    # 添加字典
    # CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    # 词的内容  词性和词频
    '''使用了静态添加的词典,故弃用动态添加'''
    # word_dict_path='../../Resources/word_Dict.txt'
    # with open(word_dict_path,encoding='utf-8') as f:
    #     dict_list=f.readlines()
    # for word in dict_list:
    #     single_wl=word.strip("\n").split(",")
    #     CustomDictionary.insert(single_wl[0],single_wl[1])
    # 使用标准分词
    wordList= HanLP.segment(raw)
    # print(wordList)
    # 保存清洗后的数据
    wordList1=str(wordList).split(',')
    # 去除词性的标签
    str_words=""
    # 需要过滤掉的词性
    word_feature=['/an','/g','/nb','/nz','/nba','/nbp','/nf','/nh','/nhd','/nhm','/ni', 
    '/nit','/nm','/nn','/nr','/nt','/nrf','/nnt','/ns','/nis','/ntc','/ntcb','/nth','/n','/vn']
    for v in wordList1[0:len(wordList1)]:
        if "/" in v:
            slope=v.index('/')
            # 先根据词性过滤掉一些词
            # if v[slope:]=='/m' or v[slope:]=='/q' or v[slope:]=='/f' or v[slope:]=='/u' or v[slope:]=='/t':
            if v[slope:] not in word_feature:
                continue
            # 添加换行符
            letter=v[1:slope]   # 截取/前面的字符串
            nature=v[slope+1:]  # 取出词性
            if '\n' in letter:
                str_words+="\n"
            else:
                letter=letter.strip()  # 去除空格
                if flag_stop == True:
                    '''去停用词'''
                    if letter not in stopwordlist: 
                        str_words+=letter+" "
                    else:
                        continue
                else:
                    str_words+=letter+" "
    return str_words


'''
读取分词文件，返回新闻列表
'''
def LoadWordList(CutWordtxt):
    contents=open(CutWordtxt,'r',encoding='utf-8').read()
    content_List=contents.splitlines()  # 分词之后的新闻内容列表
    return content_List

'''
二级新闻列表，每一条新闻是一个列表，以句号分隔
'''
def Sentences_list(content_List):
    sentence_list=[]
    for item in content_List:
        itemlist=item.split('。')
        sentence_list.append(itemlist)
    return sentence_list


'''
特征提取
'''
def TF_IDF(n_features,content_List):
    # n_features = 1000  # 1000个最重要的特征关键词
    # 因为 CountVectorizer 只是单纯的统计词频，改用TfidfVectorizer
    tf_vectorizer=TfidfVectorizer(strip_accents='unicode',
                                  max_features=n_features,
                                  stop_words='english',
                                  max_df=0.5,
                                  min_df=10)
    tf = tf_vectorizer.fit_transform(content_List)
    return tf,tf_vectorizer


# if __name__ == '__main__':
#     # raw=u"西藏藏医学院党委书记赤列旺杰在日前召开的高校联合对口支援西藏藏医学院工作协调会上介绍说"
#     raw1="自治区文化科技卫生法律和爱国爱教宣传服务“五下乡”集中示范活动在山南市扎囊县吉汝乡举行"
#     raw2="唐卡文化艺术，在党中央、国务院的亲切关怀下,中国梦是中国人的自强梦"
#     raw3="深入学习贯彻党的十九大精神"
#     # print(HanLP.segment(raw2))

#     print(HanLp_Segment(raw2,flag_stop=True))
#     raw4="唐卡 文化 艺术 党中央 国务院 关怀 中国梦 中国 唐卡"
#     print(HanLP.extractKeyword(raw4,5))
    