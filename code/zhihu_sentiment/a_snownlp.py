from snownlp import SnowNLP,sentiment
import pandas as pd
import pylab as pl
import numpy as np
from pyecharts import Bar
import json
import re
import pyhanlp
import sys
sys.path.append(r'../common/')
import file_op

pl.mpl.rcParams['font.sans-serif'] = ['SimHei']
# snownlp情感分析
def sentiment_snownlp(content_list):
    print("正在进行情感分析...")
    sentences=[]
    sentences_score=[]
    for item in content_list:
        sentences.append(item)  # 每条新闻的id
        s=SnowNLP(item)   
        score=s.sentiments
        sentences_score.append(score)  # 每条新闻的得分
    return sentences,sentences_score


# 绘制情感分析曲线
def Plt_sentiment(sentences,sentences_score):
    table = pd.DataFrame(sentences, sentences_score)  # 制成表格
    x=np.arange(len(sentences))
    pl.plot(x, sentences_score)
    pl.title(u' 情感分析')
    pl.xlabel(u'评 论 用 户')
    pl.ylabel(u'情 感 程 度')
    pl.show()

# 情感分析结果  返回每篇新闻的情感得分
def small_Bar_sentiment(sentences,sentences_score):
    scorelist=[]
    dict_score={
        "消极":0,
        "积极":0,
    }
    for item in sentences_score:
        if item>0 and item<=0.5:
            dict_score["消极"]+=1
        else:
            dict_score["积极"]+=1
        scorelist.append(round(item))
    bar=Bar("情感得分人数统计")
    bar.width=1000
    bar.add("人数",list(dict_score.keys()),list(dict_score.values()),xaxis_interval=0,xaxis_rotate=35)
    bar.render("./../../Resources/htmlfiles/sentimentbar.html")
    return dict_score,scorelist

# 对评论语料进行清洗,返回无标签的评论内容列表
def Read_comment_file(filename):
    with open(filename,'r',encoding='utf-8') as f:
        comments=f.read()
        # 去除特殊字符
        clean_txt = re.sub(r'<.*?>','',comments)
        clean_txt=re.sub(r'[> ～ _ ，？！/ // @ \[ \] 【 】# " “ ” % 。! ~ \- : { } ＂＂ ? , \. 》《 （）：]','',clean_txt)
        # 去除英文字符和数字
        clean_txt=re.sub(r'[A-Za-z\d]*','',clean_txt)
        # 对文本进行分词
        clean_list=clean_txt.split('\n')
        return clean_list




if __name__ == '__main__':
  

#####################################################################
    print("正在加载训练集...")
    # 必须传入positive.txt和negative.txt
    sentiment.train('./../../Resources/sentiment_folders/hotel/positive.txt', './../../Resources/sentiment_folders/hotel/neg.txt') # 修改
    sentiment.save('sentiment.marshal')
    
    #############################################
    # # 测试的json文件
    # filename='./../../Resources/jsonfiles/ChnSentiCorp.json' # 修改
    # type_list,content_list=file_op.readfile(filename)
    ###############################################
    # 知乎的评论内容作为测试集
    comment_file='./../../Resources/CutWordPath/sentiment_comment.txt'
    content_list=Read_comment_file(comment_file)
    ###################################################

    # 进行snownlp情感分析
    sentences,sentences_score=sentiment_snownlp(content_list)

    # 绘图,返回情感得分字典和列表
    dict_score,scorelist=small_Bar_sentiment(sentences,sentences_score) 

    print("snownlp情感分析结果统计")
    print(dict_score)

    print("测试集的结果统计")
    test_dict={
        "消极":type_list.count('0'),
        "积极":type_list.count('1')
    }
    print(test_dict)

    # # 计算准确率
    # count=0
    # type_list=list(map(int,type_list)) # 将typelist的数据转化为整型
    # for i in range(len(type_list)):
    #     if type_list[i] != scorelist[i]:
    #         count+=1
    # print("准确率为:",count/len(type_list))
   
    