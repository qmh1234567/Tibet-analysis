from snownlp import SnowNLP
import pandas as pd
import pylab as pl
import numpy as np

pl.mpl.rcParams['font.sans-serif'] = ['SimHei']

def readfile(filename):
    with open(filename,'r',encoding='utf-8') as f:
        content_list=f.readlines()
        print('读入文件成功')
    return content_list

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



if __name__ == '__main__':
    filename='./../../../Resources/CutWordPath/sentiment_cut.txt'
    content_list=readfile(filename)
    sentences,sentences_score=sentiment_snownlp(content_list[0:1000])
    Plt_sentiment(sentences,sentences_score)
    
    