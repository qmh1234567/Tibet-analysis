from pyhanlp import SafeJClass
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH
import json
# 设置路径，否则会从配置文件中寻找
# HANLP_DATA_PATH = "/home/fonttian/Data/CNLP"

"""
获取测试数据路径，位于$root/data/textClassification/sogou-mini，
根目录由配置文件指定,或者等于我们前面手动设置的HANLP_DATA_PATH。
"""
DATA_FILES_PATH = "./ChnSentiCorp"
# DATA_FILES_PATH="./weiboCorp"

NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')
ChnSentiCorp_path="./ChnSentiCorp/"  # 训练集文件目录
# weibo_path="./weiboCorp"

def train_or_load_classifier(path):
    classifier = NaiveBayesClassifier()
    classifier.train(ChnSentiCorp_path)
    return classifier


def predict(classifier, text):
    print("《%16s》\t情感极性是\t【%s】" % (text, classifier.classify(text)))


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





if __name__ == '__main__':
    classifier = train_or_load_classifier(ChnSentiCorp_path)  # 加载训练集
    # content_list=[]
    # with open("./1.txt",'r',encoding='utf-8') as f:
    #     content_list=f.read().splitlines()
    # han_list=[]
    # for item in content_list:
    #     print(item)
    #     result=classifier.classify(item)
    #     han_list.append(result)
    #     print(len(han_list))
    
    # print("hanlp情感分析结果")
    # print("消极",han_list.count('negtive'))
    # print("积极",han_list.count('positive'))

    # # 先用两句话测试一下
    predict(classifier, "这个酒店环境环境差，服务态度差 有点不太卫生，感觉不怎么样")
    predict(classifier,"下次会继续光顾")

    # 用测试集进行测试
    filename='./../../Resources/jsonfiles/sentiment_datatrain.json'
    typelist,contentlist=readfile(filename)

    hantypelist=[]
    for text in contentlist:
        hantypelist.append(classifier.classify(text))
    
    # hanlp情感分析结果
    print("hanlp情感分析结果")
    print("消极",hantypelist.count('negtive'))
    print("积极",hantypelist.count('positive'))

    hlist=[]
    for item in hantypelist:
        if item=="negtive":
            hlist.append('0')
        else:
            hlist.append('1')


    # 正确结果
    print("正确结果")
    print("消极",typelist.count('0'))
    print('积极',typelist.count('1'))

    
    # 准确率
    count=0
    for i in range(len(typelist)):
        if typelist[i] != hlist[i]:
            count+=1
    print("count=",count)
    print("准确率:",(1-count/len(typelist)))

    
    
    
    

