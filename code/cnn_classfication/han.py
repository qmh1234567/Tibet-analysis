from pyhanlp import SafeJClass
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH
import sys
sys.path.append(r'./../common/')
import file_op


# 测试数据路径
DATA_FILES_PATH = "./../../Resources/classfication_folders/train_corpus"  # 需修改


NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')

# 训练集文件目录
Sougou_path="./../../Resources/classfication_folders/train_corpus"  # 需修改


# 构建分类器
def train_or_load_classifier(path):
    classifier = NaiveBayesClassifier()
    classifier.train(Sougou_path)
    return classifier

def predict(classifier, text):
    print("《%16s》\t属于分类\t【%s】" % (text, classifier.classify(text)))


if __name__ == '__main__':

    classifier = train_or_load_classifier(Sougou_path)  # 加载训练集
     # 用测试集进行测试
    test_filename='./../../Resources/jsonfiles/fudan_test.json' # 修改
    typelist,contentlist=file_op.readfile(test_filename)

    # hanlp分类结果
    hantypelist=[] 
    for text in contentlist:
        hantypelist.append(classifier.classify(text))

    print("hanlp文本分类结果统计")
    han_dict={
        "经济":hantypelist.count('Economy'),
        "环境":hantypelist.count('Environment'),
        "政治":hantypelist.count('Politics')
    }
    print(han_dict)


    print("测试集的结果统计")
    test_dict={
        "经济":typelist.count('Economy'),
        "环境":typelist.count('Environment'),
        "政治":typelist.count('Politics')
    }
    print(test_dict)

    # 准确率
    count=0
    # print(len(typelist))
    # print(len(hantypelist))
    for i in range(len(typelist)):
        if typelist[i] != hantypelist[i]:
            count+=1   # 不正确的个数
    print("准确率:",(1-count/len(typelist)))


    

