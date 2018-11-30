from snownlp import SnowNLP
from pyhanlp import *
import sys
sys.path.append(r'../common/')
from txt_Word2Vec import Process_News


# SnowNLP库：
# words：分词
# tags：关键词
# sentiments：情感度
# pinyin：拼音
# keywords(limit)：关键词
# summary：关键句子
# sentences：语序
# tf：tf值
# idf：idf值
# s=SnowNLP(u'文青很极端，你也很偏激。。。。12年去的时候印象挺好的')
# print(s.words)
# print(s.sentiments)


'''不要过滤情感词语'''
text = u'''
能不能别把西藏说的那么龌龊 是有很多藏民不好 但是也不致于贬低当地来讽刺那些穷游的人吧
'''
s = SnowNLP(text)
m1=HanLP.segment(text)
print(m1)
print("*"*40)
m2=Process_News(text)
print(m2)
m=HanLP.extractKeyword(m2,20)
print(m)
print("*"*40)
s=s.keywords(20)
print(s)
s1=" ".join(s)
s = SnowNLP(s1)
print(s.sentiments)
print("*"*40) 
s2=" ".join(m)
print(s2)
s3 = SnowNLP(s2)
print(s3.sentiments) 

print("*"*70)
# print(s.summary(3))
# s=s.summary(3)

# print(s.keywords(6))  # [u'语言', u'自然', u'计算机'] 不能用tags输出关键字.
# print(s.summary(3))
 # 1.0
# s = SnowNLP([[u'这篇', u'文章'],
#              [u'那篇', u'论文'],
#              [u'这个']])
# print(s.tf)
# print(s.idf)
# print(s.sim([u'文章']))  # [0.3756070762985226, 0, 0]