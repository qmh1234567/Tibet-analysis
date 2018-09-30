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
# 处理后文本的保存路径
def mkdir(path):
    # 判断路径是否存在
    isExist=os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(path+"创建成功")
        return True
    else:
        pass
    print("文本正在处理中....")

# 创建停用词表
def stopwordslist(path):
    stpwds=open(path,"r",encoding='utf-8').read()
    # 将文件内容转化为列表
    stplst=stpwds.splitlines()
    return stplst

# 生成词云
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
    
# HanLp 分词  默认为去停用词  返回分好词的字符串
# raw代表一条新闻的内容  默认去停用词
def HanLp_Segment(raw,flag_stop=True):
    # 停用词列表
    stop_word_path='Resources/stopwords.txt'
    stopwordlist=stopwordslist(stop_word_path)
    # 默认分词
    # wordList=HanLP.segment(raw)
    #Hanlp分词
    NLPTokenizer=JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
    wordList= NLPTokenizer.segment(raw)
    # 保存清洗后的数据
    wordList1=str(wordList).split(',')
    # print(wordList1)
    # 去除词性的标签
    str_words=""
    for v in wordList1[0:len(wordList1)-1]:
        if "/" in v:
            slope=v.index('/')
            letter=v[1:slope]   # 截取/前面的字符串
            if v[slope:]=='/m' or v[slope:]=='/q' or v[slope:]=='/f' or v[slope:]=='/u' or v[slope:]=='/t':
                continue
            # 添加换行符
            letter=letter.strip()  # 去除空格
            '''去停用词'''
            if flag_stop == True: 
                if letter not in stopwordlist: 
                    str_words+=letter+" "   
                else:
                    continue
            else:
                str_words+=letter+" " 
    return str_words 


'''
   Hanlp 进行中文分词处理
   read_folder_path: 待处理的语料原始路径
   write_folder_path: 经数据清洗后的语料路径
'''
def Hanlp_Seg(read_folder_path,write_folder_path,stopwordlist):
    # 获取待处理根目录下的所有文件夹
    folderlist=os.listdir(read_folder_path)
    # 遍历每个文件夹 
    for folder in folderlist:
        # 某个文件夹的路径
        new_folder_path=os.path.join(read_folder_path,folder)
        # 调用mkdir()创建文件夹保存路径
        mkdir(write_folder_path+folder)
        # 处理后的文件夹保存路径
        save_folder_path=os.path.join(write_folder_path,folder)
        # 某文件夹下的全部文件集
        files = os.listdir(new_folder_path)
        j=1  # 文件计数从1开始
        # 遍历文件
        for file in files:
            # j大于文件数时终止
            if j>len(files):
                break
            # 读取原始语料
            raw = open(os.path.join(new_folder_path,file),'r',encoding='utf-8').read()
            # Hanlp分词
            # NLPTokenizer=JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
            # wordList= NLPTokenizer.segment(raw)
            str_words=HanLp_Segment(raw)
            # 处理文件操作带来的异常
            with open(os.path.join(save_folder_path,file),'w',encoding='utf-8') as f:
                f.write(str_words)
            '''生成词云''' 
            # GenWordCloud(str_words)
            # filename=os.path.splitext(file) # 获得文件名字
            # with open(os.path.join(save_folder_path,filename[0]+"Count.txt"),'w',encoding='utf-8') as f1:
            #     '''统计词频'''
            #     str_words1=str_words.replace("\n","")
                # words_dict=WordCount(str_words1)  
                # 遍历词典，写入文件
                # for k,v in words_dict.items():
                    # f1.write("%s,%d\n" % (k,v))   
            j+=1




# if __name__ == '__main__':
    # print("开始进行文本分词操作\n")
    # stop_word_path='stopwords.txt'
    # # 停用词列表
    # stopwordlist=stopwordslist(stop_word_path)
    # # 待分词的语料根目录
    # read_folder_path='news/'
    # write_folder_path='CutNews/'
    # t1=time.time()
    # # HanLP分词
    # Hanlp_Seg(read_folder_path,write_folder_path,stopwordlist)
    # t2=time.time()
    # print("完成中文文本分词"+str(t2-t1)+"秒")
    
  
        # print(dicts)
    # print(dicts{1}['content'])
  
   

  


        
