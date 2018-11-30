import json
from collections import Counter
import numpy as np
import itertools
import sys
from gensim.models import word2vec,KeyedVectors
sys.path.append(r'../common/')
from txt_Word2Vec import Process_News
from gensim.models import word2vec,KeyedVectors

class w2v_wrapper:
     def __init__(self,file_path):
        # w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.model =word2vec.Word2Vec.load(file_path)  # 加载词向量文件
        if 'unknown' not  in self.model.wv.vocab:
            unknown_vec = np.random.uniform(-0.1,0.1,size=128)  # 产生300个[-0.1,0.1)的数 从均匀分布中随机采样
            self.model.wv.vocab['unknown'] = len(self.model.wv.vocab)
            self.model.wv.vectors = np.row_stack((self.model.wv.vectors,unknown_vec))
            

'''加载数据集 
输入：cutwordfile 分词后的文件路径  jsonfile：待处理的json文件 labelfile: 处理后的标签文件路径
输出：x_text：三种类型的新闻列表 y：每条新闻的标签
备注： 第一次调用或输入改变时需要用到返回值，第二次调用则直接读取文件
'''
def load_data_and_labels(cutwordfile=None,jsonfile=None,labelfile=None):
    x_text=[]
    with open(jsonfile,"r",encoding='utf-8') as f:
        dicts=json.load(f)
        # 新闻类别
        y=[]
        # 新闻内容
        x_text=[]
        # 标签说明
        label={
            # 消极
            "0":[0,1],
            # 积极
            '1':[1,0],
        }
        print("loading data...")
        newcontent=[]
        for dict_item in dicts:
            if dict_item['type'] in label.keys() :
                key=label[dict_item['type']]
                y.append(key)
                # 直接取分词后的内容
                content = dict_item['content']
                x_text.append(content)
            else:
                continue            
    with open(cutwordfile,'w',encoding='utf-8') as f:
        for sentence in x_text:
            f.write(sentence+'\n')
    # 转化为np.array类型
    y=np.array(y)
    # 将y写入txt文件
    np.savetxt(labelfile,y)
    print("加载数据成功!")
    return [x_text,y]

# 根据word2vec建立词典
def get_text_idx(text,vocab,max_document_length):
    text_array = np.zeros([len(text), max_document_length],dtype=np.int32)
    for i,x in  enumerate(text):
        words = x.split(" ")
        for j, w in enumerate(words):  # j代表词出现的位置
            if w in vocab:
                text_array[i, j] = vocab[w].index  # 将index添加到词典中
            else :
                text_array[i, j] = vocab['unknown']
    return text_array


# 针对一个数据集生成一个批迭代器  生成批次数据
# 使用生成器生成 num_epochsx num_batches_per_epoch 个批次的数据用于训练
def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)
    data_size=len(data)
    # 每个epoch 的num_batch
    num_batches_per_epoch=int(len(data)/batch_size)+1
    for epoch in range(num_epochs):
        # shuffle_indices 的len为 data_size 随机元素组成的列表
        # 在每次迭代中随机打乱数据
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data        
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,data_size)
            yield shuffled_data[start_index:end_index]


# if __name__ == '__main__':
#     txtfile='../../Resources/sentiment_datatrain.json'
#     labelfile='../../Resources/sentiment_label.txt'
#     cutwordfile='../../Resources/sentiment_cut.txt'
#     load_data_and_labels(cutwordfile,txtfile,labelfile)
    