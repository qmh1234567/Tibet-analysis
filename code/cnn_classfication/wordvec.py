import sys
from data_helpers import load_data_and_labels
sys.path.append(r'../common/')
from txt_Word2Vec import Tibet_Word2Vec,Wordlist_View
import tensorflow as tf
from gensim.models import word2vec,KeyedVectors
from txt_preprocess import LoadWordList
from pyhanlp import *
import numpy as np

# tf.flags.DEFINE_string("jsonfile", "./../../Resources/jsonfiles/data_train.json", "Data source for the json file.")
# tf.flags.DEFINE_string("cutwordfile", "./../../Resources/CutWordPath/data_train.txt", "Data source for the cutword save file.")
# tf.flags.DEFINE_string("labelfile", "./../../Resources/labels/data_train_label.txt", "label save file")
# # tf.flags.DEFINE_string("cutkeywordfile", "./../../Resources/CutWordPath/data_train_keyword.txt", "label save file")
# tf.flags.DEFINE_string("Binaryfile", "./../../Resources/Binaryfiles/datatrain_vector.bin", "binary file")

#FLAGS保存命令行参数的数据
FLAGS = tf.flags.FLAGS




# 绘制与三种类别最相似的top15单词位置
def plt_Word2vec(Binaryfile):
    # 加载词向量的二进制文件
    model=word2vec.Word2Vec.load(Binaryfile)
    # 输出与某个词相近的词
    wordlist=['政治','社会','文化']
    Wordlist_View(wordlist,model,15)


# 提取关键词
def Hanlp_keyword(CutWordtxt,CutKeyWordtxt,KeyWord_Count=200):
    # 读取分词后的文件
    content_List=LoadWordList(CutWordtxt)
    # contents=Read_file(jsonfile,CutWordtxt,flag_stop=True)
    # 统计最大长度的新闻
    max_len=max(len(x) for x in content_List)
    print("max_len=",max_len)
    # 提取每篇文章的关键字
    Keywordlists=[]
    for single_new in content_List:
        if len(single_new)<KeyWord_Count:
            KeyWord_Count=100
        keywordList=HanLP.extractKeyword(single_new,KeyWord_Count)
        Keywordlists.append(keywordList)
    with open(CutKeyWordtxt,'w',encoding='utf-8') as f:
        for itemlist in Keywordlists:
            for strword in itemlist:
                f.write(strword+" ")
            f.write("\n")



# 生成word2vec词向量的二进制文件 
def generate_word2vec():
    # 第一次调用
    # print("loading data")
    x_text,y=load_data_and_labels(FLAGS.cutwordfile,FLAGS.jsonfile,FLAGS.labelfile)
    model=Tibet_Word2Vec(FLAGS.cutwordfile,FLAGS.Binaryfile)
    # 画词向量图
    plt_Word2vec(FLAGS.Binaryfile)
    


if __name__ == '__main__':
    generate_word2vec()

   


    
