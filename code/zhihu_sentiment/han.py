from pyhanlp import SafeJClass
import zipfile
import os
from pyhanlp.static import download, remove_file, HANLP_DATA_PATH
import sys
sys.path.append(r'../common/')
import file_op


# 测试数据路径
DATA_FILES_PATH = "./../../Resources/sentiment_folders/ChnSentiCorp"  # 需修改


NaiveBayesClassifier = SafeJClass('com.hankcs.hanlp.classification.classifiers.NaiveBayesClassifier')
IOUtil = SafeJClass('com.hankcs.hanlp.corpus.io.IOUtil')

# 训练集文件目录
ChnSentiCorp_path="./../../Resources/sentiment_folders/ChnSentiCorp"  # 需修改


# 构建分类器
def train_or_load_classifier(path):
    classifier = NaiveBayesClassifier()
    classifier.train(ChnSentiCorp_path)
    return classifier




if __name__ == '__main__':

    classifier = train_or_load_classifier(ChnSentiCorp_path)  # 加载训练集

    ###############################################
    # 用测试集进行测试
    # test_filename='./../../Resources/jsonfiles/hotelCorp.json' # 修改
    # typelist,contentlist=file_op.readfile(test_filename)
    ###############################################
    # 知乎的评论内容作为测试集
    comment_file='./../../Resources/CutWordPath/sentiment_comment.txt'
    content_list=Read_comment_file(comment_file)
    ###############################################


    # hanlp情感分析结果
    hantypelist=[] 
    for text in contentlist:
        hantypelist.append(classifier.classify(text))

    print("hanlp情感分析结果统计")
    han_dict={
        "消极":hantypelist.count('negtive'),
        "积极":hantypelist.count('positive')
    }
    print(han_dict)
    hlist=[]   # hanlp情感分析结果列表
    for item in hantypelist:
        if item=="negtive":
            hlist.append('0')
        else:
            hlist.append('1')
    
    print("测试集的结果统计")
    test_dict={
        "消极":typelist.count('0'),
        "积极":typelist.count('1')
    }
    print(test_dict)

    # # 准确率
    # count=0
    # for i in range(len(typelist)):
    #     if typelist[i] != hlist[i]:
    #         count+=1   # 不正确的个数
    # print("准确率:",(1-count/len(typelist)))

    
    
    

