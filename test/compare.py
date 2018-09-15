from pyhanlp import *
import jieba
from jpype import *
import myhanlp
import time
import requests
# 读取文件
def File_open():
    f=open("news/government/news1.txt","r",encoding='utf-8')
    line=f.readline()
    print(type(line))
    list1=[]
    while line:
        line=line.strip('\n')     # 去除每一行的换行符
        list1.append(line)
        line=f.readline()
    f.close()
    list2=list(filter(None,list1))  # 去除列表中的空字符
    return list2


def HanlpSeg(list1):
    t1=time.time()
    print("-"*70)
    for line in list1:
        seg=HanLP.segment(line)
        print(seg)
    t2=time.time()
    print("hanlp time:",t2-t1)

def JiebaCut(list1):
    t1=time.time()
    print("-"*70)
    for line in list1:
        jb=jieba.cut(line)
        print("/".join(jb))
    t2=time.time()
    print("jieba:",t2-t1)


if __name__ == '__main__':
    list1=File_open()
    HanlpSeg(list1)
    JiebaCut(list1)
    myhanlp.Define_segment()
    
    


