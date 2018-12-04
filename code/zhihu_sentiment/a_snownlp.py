from snownlp import SnowNLP
import pandas as pd
import pylab as pl
import numpy as np
from pyecharts import Bar
import json
pl.mpl.rcParams['font.sans-serif'] = ['SimHei']


def readfile(filename):
    with open(filename,'r',encoding='utf-8') as f:
        dict_sentiment=json.load(f)
        content_list=[]
        type_list=[]
        for dict1 in dict_sentiment:     
            content_list.append(dict1['content'])
            type_list.append(dict1['type'])
        print('读入文件成功')
    return type_list,content_list

# 情感分析
def sentiment_snownlp(content_list):
    print("正在进行情感分析...")
    sentences=[]
    sentences_score=[]
    for item in content_list:
        sentences.append(item)
        s=SnowNLP(item)
        score=s.sentiments
        sentences_score.append(score)
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


def Bar_sentiment(sentences,sentences_score):
    name=np.arange(len(sentences))
    num=sentences_score
    bar=Bar("情感分析")
    bar.use_theme("roma")
    bar.width=1000
    bar.height=340
    bar.add(
        "情感得分",
        name,
        num,
        xaxis_interval=0,
        xaxis_rotate=45,
    is_datazoom_show=True,
    )
    bar.render("./1.html")

def small_Bar_sentiment(sentences,sentences_score):
    dict_score={
        "消极":0,
        "积极":0,
    }
    for item in sentences_score:
        if item>0 and item<=0.5:
            dict_score["消极"]+=1
        else:
            dict_score["积极"]+=1
    bar=Bar("情感得分人数统计")
    bar.width=1000
    bar.add("人数",list(dict_score.keys()),list(dict_score.values()),xaxis_interval=0,xaxis_rotate=35)
    bar.render("./../../Resources/htmlfiles/sentimentbar.html")





if __name__ == '__main__':
    filename='./../../Resources/jsonfiles/sentiment_datatrain.json'
    type_list,content_list=readfile(filename)
    sentences,sentences_score=sentiment_snownlp(content_list)
    print("消极",type_list.count('0'))
    print('积极',type_list.count('1'))
    # Bar_sentiment(sentences,sentences_score)
    small_Bar_sentiment(sentences,sentences_score)
    # Plt_sentiment(sentences,sentences_score)
   
    