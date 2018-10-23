from jpype import *
import os
from pyhanlp import *
import time
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import pylab
from PIL import Image
import numpy as np

'''创建停用词表'''
def stopwordslist(path):
    stpwds=open(path,"r",encoding='utf-8').read()
    # 将文件内容转化为列表
    stplst=stpwds.splitlines()
    return stplst

'''生成词云: str_words为字符串 生成词云图片'''
def GenWordCloud(str_words):
    image=plt.imread('img.jpg')
    wc=WordCloud(background_color='white',
                font_path="Resource/fangsong_GB2312.ttf", # 设置字体格式，不设置显示不了中文
                max_words=2000,
                mask=image, # 词云形状
                )
    wc.generate(str_words)
    image_colors=ImageColorGenerator(image)
    wc.recolor(color_func=image_colors)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
    # wc.to_file("1.jpg")

'''
HanLp 分词函数
raw代表一条新闻内容 flag_stop=True表示去停用词
str_words：分词后的字符串
'''
def HanLp_Segment(raw,flag_stop=True):
    # 停用词列表
    stop_word_path='Resources/stopwords.txt'
    stopwordlist=stopwordslist(stop_word_path)
    #Hanlp分词
    NLPTokenizer=JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
    wordList= NLPTokenizer.segment(raw)
    # 保存清洗后的数据
    wordList1=str(wordList).split(',')
    # 去除词性的标签
    str_words=""
    for v in wordList1[0:len(wordList1)]:
        if "/" in v:
            slope=v.index('/')
            # 先根据词性过滤掉一些词
            if v[slope:]=='/m' or v[slope:]=='/q' or v[slope:]=='/f' or v[slope:]=='/u' or v[slope:]=='/t':
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
                        # letter.replace('\n','')
                        # letter.replace('\r','')
                    else:
                        continue
                else:
                    str_words+=letter+" "
    return str_words

